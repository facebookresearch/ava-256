# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Train an autoencoder."""

import argparse
import itertools
import logging
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Union

import einops
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from fvcore.common.config import CfgNode as CN
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from data.ava_dataset import MultiCaptureDataset as AvaMultiCaptureDataset
from data.ava_dataset import none_collate_fn
from data.utils import get_framelist_neuttex_and_neutvert
from losses import mean_ell_1
from models.bottlenecks.vae import kl_loss_stable
from utils import get_autoencoder, load_checkpoint, render_img, tocuda, train_csv_loader

sys.dont_write_bytecode = True


class HostnameFilter(logging.Filter):
    hostname = platform.node()

    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True


root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.addFilter(HostnameFilter())
formatter = logging.Formatter("%(asctime)s %(hostname)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

FRONTAL_CAMERAS = ["401168", "401875", "402040", "401031"]

# TODO(julieta) see if this actually does anything
torch.backends.cudnn.benchmark = True  # gotta go fast!


def gen_optimizer(
    net: torch.nn.Module,
    optim_type: str,
    batchsize: int,
    rank: int,
    learning_rate: float,
    tensorboard_logger=None,
):
    params = filter(lambda x: x.requires_grad, itertools.chain(net.parameters()))
    if optim_type == "adam":
        opt = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))
    elif optim_type == "sgd":
        opt = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    elif optim_type == "adamw":
        opt = torch.optim.AdamW(params, lr=learning_rate, betas=(0.9, 0.999))
    else:
        raise ValueError(f"Unsupported optimizer: {optim_type}")

    if tensorboard_logger and rank == 0:
        tb_hyperparams = {
            "minibatchsize": batchsize,
            "learningrate": learning_rate,
            "optimizer": optim_type,
        }

        tensorboard_logger.add_hparams(tb_hyperparams, {"hp_metric": 1.0}, run_name=".")
    return params, opt


def setup(local_rank, world_rank, world_size, master_address: str, master_port: str):
    logging.info(f"{local_rank=} {world_rank=} {world_size=} {master_address=}, {master_port=}")
    print(f"{local_rank=} {world_rank=} {world_size=} {master_address=}, {master_port=}")

    # # devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    # print(f"{devices=}")

    # if local_rank is not None:
    torch.cuda.set_device(local_rank)

    # if world_rank is None:
    #     world_rank = local_rank

    os.environ["MASTER_ADDR"] = master_address
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group("nccl", rank=world_rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare(rank, world_size, train_params):
    base_dir = train_params.dataset_dir
    train_dirs = None
    train_captures = None

    # Load dataset
    train_captures, train_dirs = train_csv_loader(base_dir, train_params.data_csv, train_params.nids)
    dataset = AvaMultiCaptureDataset(train_captures, train_dirs, downsample=train_params.downsample)

    starttime = time.time()

    maxiter = train_params.maxiter  # that ought to be enough for anyone
    logging.info(" maxiter :  {}, batchsize: {}".format(maxiter, train_params.batchsize))

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_params.batchsize,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        num_workers=train_params.num_workers,
        collate_fn=none_collate_fn,
    )

    # Store neut avgtex and neut vert from all ids for x-id check
    all_neut_avgtex_vert = []

    for directory in train_dirs:
        _, neut_avgtex, neut_vert = get_framelist_neuttex_and_neutvert(directory)

        neut_avgtex = (neut_avgtex - dataset.texmean) / dataset.texstd
        neut_verts = (neut_vert - dataset.vertmean) / dataset.vertstd
        all_neut_avgtex_vert.append({"neut_avgtex": torch.tensor(neut_avgtex), "neut_verts": torch.tensor(neut_verts)})

    # Driver data for x-id
    driver_dataset = AvaMultiCaptureDataset(
        train_captures, train_dirs, downsample=train_params.downsample, cameras_specified=FRONTAL_CAMERAS
    )

    driver_sampler = DistributedSampler(
        driver_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )

    driver_dataloader = torch.utils.data.DataLoader(
        driver_dataset,
        batch_size=1,
        sampler=driver_sampler,
        shuffle=False,
        drop_last=True,
        num_workers=1,
        collate_fn=none_collate_fn,
    )

    logging.info("Datasets instantiated ({:.2f} s)".format(time.time() - starttime))

    return dataset, all_neut_avgtex_vert, dataloader, driver_dataloader


def xid_eval(model, driver_dataiter, all_neut_avgtex_vert, config, output_set, outpath, rank, iternum):
    starttime = time.time()

    indices_subjects = random.sample(range(0, len(all_neut_avgtex_vert)), config.progress.cross_id_n_subjects)
    indices_subjects.sort()
    model.eval()

    with torch.no_grad():
        driver = next(driver_dataiter)
        while driver is None:
            driver = next(driver_dataiter)

        cudadriver: Dict[str, Union[torch.Tensor, int, str]] = tocuda(driver)

        gt = cudadriver["image"].detach().cpu().numpy()
        gt = einops.rearrange(gt, "1 c h w -> h w c")
        renderImages_xid = []
        renderImages_xid.append(gt)

        running_avg_scale = False
        gt_geo = None
        residuals_weight = 1.0

        output_driver = model(
            cudadriver["camrot"],
            cudadriver["campos"],
            cudadriver["focal"],
            cudadriver["princpt"],
            cudadriver["modelmatrix"],
            cudadriver["avgtex"],
            cudadriver["verts"],
            cudadriver["neut_avgtex"],
            cudadriver["neut_verts"],
            cudadriver["neut_avgtex"],
            cudadriver["neut_verts"],
            cudadriver["pixelcoords"],
            cudadriver["idindex"],
            cudadriver["camindex"],
            running_avg_scale=running_avg_scale,
            gt_geo=gt_geo,
            residuals_weight=residuals_weight,
            output_set=output_set,
        )

        rgb_driver = output_driver["irgbrec"].detach().cpu().numpy()
        rgb_driver = einops.rearrange(rgb_driver, "1 c h w -> h w c")
        del output_driver
        renderImages_xid.append(rgb_driver)

        for i in indices_subjects:
            if i == int(cudadriver["idindex"][0]):
                continue
            cudadriven: Dict[str, Union[torch.Tensor, int, str]] = tocuda(all_neut_avgtex_vert[i])

            output_driven = model(
                cudadriver["camrot"],
                cudadriver["campos"],
                cudadriver["focal"],
                cudadriver["princpt"],
                cudadriver["modelmatrix"],
                cudadriver["avgtex"],
                cudadriver["verts"],
                cudadriver["neut_avgtex"],
                cudadriver["neut_verts"],
                torch.unsqueeze(cudadriven["neut_avgtex"], 0),
                torch.unsqueeze(cudadriven["neut_verts"], 0),
                cudadriver["pixelcoords"],
                cudadriver["idindex"],
                cudadriver["camindex"],
                running_avg_scale=running_avg_scale,
                gt_geo=gt_geo,
                residuals_weight=residuals_weight,
                output_set=output_set,
            )
            rgb_driven = output_driven["irgbrec"].detach().cpu().numpy()
            rgb_driven = einops.rearrange(rgb_driven, "1 c h w -> h w c")
            del output_driven
            renderImages_xid.append(rgb_driven)
            del cudadriven
    del cudadriver
    if rank == 0:
        render_img([renderImages_xid], f"{outpath}/x-id/progress_{iternum}.png")

    print(f"Cross ID viz took {time.time() - starttime}")


def main(rank, config, args):
    """
    Rank is normally set automatically by mp.spawn()
    """

    if args.world_rank is None:
        # World rank was not set because we ran this outside of slurm, just get local rank
        args.world_rank = rank
    else:
        # World rank is set to node_id * ngpus_per_node, so we have to add the local rank
        args.world_rank = args.world_rank + rank

    logging.info(f"{rank=}, {args.world_size=}")

    setup(rank, args.world_rank, args.world_size, args.masteraddress, args.masterport)

    train_params = config.train

    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    outpath = os.path.join(current_dir, config.progress.output_path)
    os.makedirs(f"{outpath}/x-id", exist_ok=True)

    tensorboard_logger = None
    if config.progress.tensorboard.logdir is not None and args.world_rank == 0:
        tensorboard_logdir = config.progress.output_path + "/" + config.progress.tensorboard.logdir
        logging.info(f"Creating tensorboard output at {tensorboard_logdir}")
        tensorboard_logger = SummaryWriter(tensorboard_logdir)

    dataset, all_neut_avgtex_vert, dataloader, driver_dataloader = prepare(
        args.world_rank, args.world_size, train_params
    )

    starttime = time.time()
    assetpath = Path(__file__).parent / "assets"
    ae = get_autoencoder(dataset, assetpath=str(assetpath))

    # device = torch.cuda.current_device()

    if train_params.checkpoint:
        logging.info(f"Loading checkpoint from {train_params.checkpoint}")
        ae = load_checkpoint(ae, train_params.checkpoint)
    ae = ae.train().to("cuda")

    vertmean, vertstd = torch.Tensor(dataset.vertmean).to("cuda"), dataset.vertstd

    model = DDP(ae, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    iternum = 0
    if train_params.checkpoint:
        # TODO(julieta) this is not a very robust way to determine iteration number...
        # TODO(julieta) decide what to do about lr scheduler
        numbers = re.findall(r"\d+", train_params.checkpoint)
        if not numbers:
            raise ValueError(f"Checkpoint given but it does not contain an iteration number: {train_params.checkpoint}")
        iternum = int(numbers[-1])

    initial_lr = train_params.init_learning_rate

    if iternum != 0:
        # NOTE(julieta) this is not technically correct, find a better way to deal with interrupted iternum
        initial_lr = train_params.init_learning_rate * train_params.gamma

    optim_type = "adam"
    _, optim = gen_optimizer(
        model,
        optim_type,
        train_params.batchsize,
        rank,
        initial_lr,
        tensorboard_logger,
    )

    scheduler = lr_scheduler.StepLR(optim, step_size=train_params.lr_scheduler_iter, gamma=train_params.gamma)

    loss_weights: Dict[str, float] = {
        "irgbl1": train_params.losses.irgbl1,
        "vertl1": train_params.losses.vertl1,
        "kldiv": train_params.losses.kldiv,
        "primvolsum": train_params.losses.primvolsum,
    }

    logging.info("Optimizer instantiated ({:.2f} s)".format(time.time() - starttime))

    # train
    starttime = time.time()

    # Total experiment time
    lstart = time.time()

    output_set = set(train_params.output_set)
    logging.info("OUTPUT SET :{}".format(output_set))

    driver_dataiter = iter(driver_dataloader)

    for _ in range(train_params.num_epochs):
        for data in dataloader:
            if data is None:
                continue

            iter_start_time = time.time()
            cudadata: Dict[str, Union[torch.Tensor, int, str]] = tocuda(data)

            # TODO(julieta) control these by cli/config arguments
            running_avg_scale = False
            gt_geo = None
            residuals_weight = 1.0
            if iternum < 100:
                running_avg_scale = True
                gt_geo = cudadata["verts"]
                residuals_weight = 0.0

            output = model(
                cudadata["camrot"],
                cudadata["campos"],
                cudadata["focal"],
                cudadata["princpt"],
                cudadata["modelmatrix"],
                cudadata["avgtex"],
                cudadata["verts"],
                cudadata["neut_avgtex"],
                cudadata["neut_verts"],
                cudadata["neut_avgtex"],
                cudadata["neut_verts"],
                cudadata["pixelcoords"],
                cudadata["idindex"],
                cudadata["camindex"],
                running_avg_scale=running_avg_scale,
                # NOTE(julieta) These control the behaviour of the forward pass, and passing them makes the optimization
                # easier. This can be tricky to get your head around, but is crucial to understand how the optimization
                # of so many primitives manages to converge.
                gt_geo=gt_geo,
                residuals_weight=residuals_weight,
                output_set=output_set,
            )

            # TODO(julieta) make an enum for these losses
            losses: Dict[str, torch.Tensor] = {}

            if "irgbl1" in loss_weights:
                # NOTE(julieta) both are unnormalized already
                losses["irgbl1"] = mean_ell_1(output["irgbrec"], cudadata["image"])

            if "vertl1" in loss_weights:
                losses["vertl1"] = mean_ell_1(output["verts"], cudadata["verts"] * vertstd + vertmean)

            if "primvolsum" in loss_weights:
                primscale = output["primscale"]
                losses["primvolsum"] = torch.sum(torch.prod(1.0 / primscale, dim=-1), dim=-1)

            if "kldiv" in loss_weights:
                losses["kldiv"] = kl_loss_stable(output["expr_mu"], output["expr_logstd"])

            if not losses:
                raise ValueError("No losses were computed. We can't train like that!")

            # compute final loss
            loss = sum(
                [
                    loss_weights[k]
                    * (torch.sum(v[0]) / torch.sum(v[1]).clamp(min=1e-6) if isinstance(v, tuple) else torch.mean(v))
                    for k, v in losses.items()
                ]
            )
            optim.zero_grad()
            loss.backward()

            params = [p for pg in optim.param_groups for p in pg["params"]]

            for p in params:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad.data[torch.isnan(p.grad.data)] = 0
                    p.grad.data[torch.isinf(p.grad.data)] = 0

            torch.nn.utils.clip_grad_norm_(model.parameters(), train_params.clip)
            optim.step()

            # NOTE(julieta) We compute iter total time anyway -- no extra syncing needed, there is already an implicit
            # sync during `backward`
            iter_total_time = time.time() - iter_start_time

            # print progress
            if (iternum < 10000 and iternum % 100 == 0) or iternum % 1000 == 0:
                renderImages = []
                del output["verts"]
                for b in range(cudadata["camrot"].shape[0]):
                    gt = cudadata["image"][b].detach().cpu().numpy()
                    gt = einops.rearrange(gt, "c h w -> h w c")

                    rgb_orig = output["irgbrec"][b].detach().cpu().numpy()
                    rgb_orig = einops.rearrange(rgb_orig, "c h w -> h w c")
                    renderImages.append([gt, rgb_orig, (gt - rgb_orig) ** 2 * 10])

                if args.world_rank == 0:
                    render_img(renderImages, f"{outpath}/progress_{iternum}.png")

                # cross id generation
                if config.progress.cross_id and args.world_rank == 0:
                    xid_eval(model, driver_dataiter, all_neut_avgtex_vert, config, output_set, outpath, rank, iternum)
                    model.train()

            save_checkpoints = False
            if iternum < 10_000:
                # to account for early loss explosions
                if iternum % 2_000 == 0:
                    save_checkpoints = True
            else:
                if iternum % 20_000 == 0:
                    save_checkpoints = True

            if save_checkpoints and args.world_rank == 0:
                # Save model state
                fname_params = "{}/aeparams.pt".format(outpath)
                fname_params_iter = "{}/aeparams_{:06d}.pt".format(outpath, iternum)
                logging.warning(f"rank {args.world_rank} save checkpoint to outpath {fname_params}")
                logging.warning(f"rank {args.world_rank} save checkpoint to outpath {fname_params_iter}")
                torch.save(model.module.state_dict(), fname_params)
                torch.save(model.module.state_dict(), fname_params_iter)

                # Save optimizer as well
                fname_optim = "{}/optimparams.pt".format(outpath)
                fname_optim_iter = "{}/optimparams_{:06d}.pt".format(outpath, iternum)
                logging.warning(f"rank {args.world_rank} save checkpoint to outpath {fname_optim}")
                logging.warning(f"rank {args.world_rank} save checkpoint to outpath {fname_optim_iter}")
                torch.save(optim.state_dict(), fname_optim)
                torch.save(optim.state_dict(), fname_optim_iter)

            del cudadata

            if not args.nodisplayloss:
                logging.info(
                    "Rank {:>3d} Iteration {} loss = {:.4f}, ".format(args.world_rank, iternum, float(loss.item()))
                    + ", ".join(
                        [
                            "{} = {:.4f}".format(
                                k,
                                float(
                                    torch.sum(v[0]) / torch.sum(v[1].clamp(min=1e-6))
                                    if isinstance(v, tuple)
                                    else torch.mean(v)
                                ),
                            )
                            for k, v in losses.items()
                        ]
                    )
                    + f" time: {iter_total_time:.3f} s"
                )

            # Tensorboard Logging
            if tensorboard_logger is not None and iternum % config.progress.tensorboard.log_freq == 0:
                tensorboard_logger.add_scalar("Total Loss", float(loss.item()), iternum)

                tb_loss_stats = {}
                for k, v in losses.items():
                    if v.ndim == 0:
                        tb_loss_stats[k] = v
                    else:
                        tb_loss_stats[k] = torch.sum(v)

                for k, v in losses.items():
                    if v.ndim == 0:
                        tensorboard_logger.add_scalar("loss/" + k, v, iternum)
                    else:
                        tensorboard_logger.add_scalar("loss/" + k, torch.sum(v), iternum)

                # try:
                #     tensorboard_logger.add_image("progress", einops.rearrange(imgout, 'h w c -> c h w'), iternum)
                # except:
                #     raise ValueError("Tensorboard cannot log images because it is None.")
            maxiter = train_params.maxiter  # that ought to be enough for anyone

            if iternum >= maxiter:
                logging.info(f"Stopping training due to max iter limit, rank {rank} curr iter {iternum} / {maxiter}")
                lend = time.time()
                totaltime = lend - lstart
                times = {"totaltime": totaltime, "maxiter": iternum}
                np.save(f"{outpath}/timesinfo_r{rank}", times, allow_pickle=True)

                logging.info(
                    "Rank {} Iteration {} loss = {:.5f}, ".format(rank, iternum, float(loss.item()))
                    + ", ".join(
                        [
                            "{} = {:.5f}".format(
                                k,
                                float(
                                    torch.sum(v[0]) / torch.sum(v[1].clamp(min=1e-6))
                                    if isinstance(v, tuple)
                                    else torch.mean(v)
                                ),
                            )
                            for k, v in losses.items()
                        ]
                    )
                )

                break

            if iternum < train_params.lr_scheduler_iter + 1:
                scheduler.step()

            iternum += 1

    # cleanup
    cleanup()


if __name__ == "__main__":
    __spec__ = None  # to use ipdb

    # parse arguments
    parser = argparse.ArgumentParser(description="Train an autoencoder")

    # Can overwrite some of these parameter to, eg, train over multiple hosts
    parser.add_argument("--masteraddress", type=str, default="localhost", help="master node address (hostname or ip)")
    parser.add_argument("--masterport", type=str, default="43321", help="master node network port")
    parser.add_argument("--nodisplayloss", action="store_true", help="logging loss value every iteration")

    # TODO(julieta) get rid of this, there should only be one dataset in the OSS release
    parser.add_argument("--idfilepath", type=str, default=None, help="file of id list for training or evaluation")
    parser.add_argument("--config", default="configs/config.yaml", type=str, help="config yaml file")
    parser.add_argument("--opts", default=[], type=str, nargs="+")

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = CN(yaml.load(file, Loader=yaml.UnsafeLoader))

    config.merge_from_list(args.opts)

    # Validate config
    if config.progress.cross_id:
        assert (
            config.progress.cross_id_n_subjects < config.train.nids
        ), "number of subjects for cross id must be < number of subjects in the dataset"

    train_params = config.train

    # Update worldsize with number of GPUs
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        logging.info(f"Using {ngpus_per_node} GPUs per node.")
    else:
        ngpus_per_node = 1

    world_rank = os.getenv("SLURM_PROCID", None)
    world_size = os.getenv("SLURM_NTASKS", None)

    if world_rank is None and world_size is None:
        # Running manually, no slurm variables set
        args.world_rank = None
        args.world_size = ngpus_per_node
        mp.spawn(main, args=(config, args), nprocs=ngpus_per_node)
        # main(0, config, args)
    else:
        # Running things on a slurm cluster with sbatch sbatch.sh
        args.world_rank = int(world_rank) * ngpus_per_node
        args.world_size = int(world_size) * ngpus_per_node
        mp.spawn(main, args=(config, args), nprocs=ngpus_per_node)
