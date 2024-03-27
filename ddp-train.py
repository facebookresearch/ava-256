"""Train an autoencoder."""

import argparse
import hashlib as hlib
import importlib
import importlib.util
import itertools
import logging
import os
import pathlib
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Union

import einops
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from fvcore.common.config import CfgNode as CN
from torch.utils.tensorboard import SummaryWriter

from data.ava_dataset import MultiCaptureDataset as AvaMultiCaptureDataset
from data.ava_dataset import none_collate_fn
from data.utils import MugsyCapture, get_framelist_neuttex_and_neutvert
from losses import mean_ell_1
from models.bottlenecks.vae import kl_loss_stable
from progress_writer import Progress
from utils import get_autoencoder, load_checkpoint, render_img, train_csv_loader

sys.dont_write_bytecode = True

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
    worldsize: int = 1,
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
            "globalbatchsize": batchsize * worldsize,
            "learningrate": learning_rate,
            "optimizer": optim_type,
        }

        tensorboard_logger.add_hparams(tb_hyperparams, {"hp_metric": 1.0}, run_name=".")
    return params, opt


def import_module(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tocuda(d: Union[torch.Tensor, Dict, List]) -> Union[torch.Tensor, Dict, List]:
    if isinstance(d, torch.Tensor):
        return d.to("cuda")
    elif isinstance(d, dict):
        return {k: tocuda(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [tocuda(v) for v in d]
    else:
        return d


def check_quorum(token, allmembers):
    size = dist.get_world_size()
    dist.all_reduce(token, op=dist.ReduceOp.SUM, group=allmembers)
    ResetFlag = False
    if token[0] != size:
        ResetFlag = True
        # logging.warning(f"at least one of the ranks is seeing unstable loss value(s)")
    return ResetFlag


if __name__ == "__main__":
    __spec__ = None  # to use ipdb

    # parse arguments
    parser = argparse.ArgumentParser(description="Train a codec avatar universal decoder")

    parser.add_argument(
        "--noprogress",
        action="store_true",
        help="don't output training progress images",
    )
    parser.add_argument("--rank", type=int, default=0, help="process rank in distributed training")
    parser.add_argument(
        "--worldsize",
        type=int,
        default=1,
        help="the number of processes for distributed training",
    )
    parser.add_argument("--masterip", type=str, default="localhost", help="master node ip address")
    parser.add_argument("--masterport", type=str, default="43321", help="master node network port")
    parser.add_argument(
        "--nodisplayloss",
        action="store_true",
        help="logging loss value every iteration",
    )

    # TODO(julieta) get rid of this, there should only be one dataset in the OSS release
    parser.add_argument(
        "--ids2use",
        type=int,
        default=-1,
        help="the number of processes for distributed training",
    )
    parser.add_argument(
        "--idfilepath",
        type=str,
        default=None,
        help="file of id list for training or evaluation",
    )
    parser.add_argument("--config", default="config.yaml", type=str, help="config yaml file")
    parser.add_argument("--opts", default=[], type=str, nargs="+")

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = CN(yaml.load(file, Loader=yaml.UnsafeLoader))

    config.merge_from_list(args.opts)

    if config.progress.cross_id:
        if config.train.nids < config.progress.cross_id_n_subjects:
            raise ValueError(
                f"Cannot do cross evaluation on {config.progress.cross_id_n_subjects} subjects, there are only up to {config.train.nids} captures available"
            )

    print(config)

    train_params = config.train

    # TODO(julieta) remove all references to SLURM variables
    args.worldsize = int(os.environ.get("SLURM_NTASKS", args.worldsize))
    args.rank = int(os.environ.get("SLURM_PROCID", args.rank))

    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    outpath = os.environ.get(
        "RSC_RUN_DIR", os.path.join(current_dir, config.progress.output_path)
    )  # RSC_EXP_RUN_BASE_DIR/SLURM_NODEID/SLURM_LOCALID
    os.makedirs(f"{outpath}/x-id", exist_ok=True)
    # os.makedirs(outpath, exist_ok=True)

    logpath = "{}/log-r{}.txt".format(outpath, args.rank)

    # if path exists append automatically
    # logging.info(f"Python {sys.version}")
    # logging.info(f"PyTorch {torch.__version__}")
    # logging.info(" ".join(sys.argv))
    # logging.info(f"Output path: {outpath}")

    # logging.info(" ==================================================== ")
    # logging.info(" args.rank: {}, args: {}".format(args.rank, args))
    # v_devices = os.getenv("CUDA_VISIBLE_DEVICES", "NOTFOUND")
    # logging.info(" ==================================================== ")
    # logging.info(
    #     " At rank {} DEVICE COUNT : {} -- CUDA VISIBLE DEVICES {}".format(
    #         args.rank, torch.cuda.device_count(), v_devices
    #     )
    # )
    # logging.info(" ==================================================== ")

    cuda_device_count = torch.cuda.device_count()

    tensorboard_logger = None
    if config.progress.tensorboard.logdir is not None and args.rank == 0:
        tensorboard_logdir = config.progress.output_path + "/" + config.progress.tensorboard.logdir
        logging.info(f"Creating tensorboard output at {tensorboard_logdir}")
        tensorboard_logger = SummaryWriter(tensorboard_logdir)

    # logging.info("@@@@@@@@@@@@@@@ JOB CONFIG:")
    # logging.info(args)
    # logging.info("@@@@@@@@@@@@@@@ END OF JOB CONFIG \n")

    # logging.info("@@@@@@@@@@@@@@@ OS ENV VARIABLES (from sbatch):")
    # for k, v in os.environ.items():
    #     logging.info(f"{k}:{v}")
    # logging.info("@@@@@@@@@@@@@@@ END OF OS ENV VARIABLES :")

    # logging.info(f"master ip {args.masterip}")

    rank = args.rank
    worldsize = args.worldsize

    disturl = f"tcp://{args.masterip}:{args.masterport}"
    logging.info(f" DIST URL : {disturl}")

    # TODO(julieta) get the number of workers from the command line
    numworkers = train_params.num_workers

    enableddp = False

    if args.worldsize > 1:
        logging.info(f" INIT DDP RANK {args.rank}  WSIZE {args.worldsize}  URL {disturl}")
        os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
        dist.init_process_group(
            backend="nccl",
            init_method=disturl,
            world_size=args.worldsize,
            rank=args.rank,
        )
        logging.info(f" distributed training group is initialized at rank : {args.rank}")

        allmembers = None
        loss_quorum = None
        allmembers = dist.new_group(range(worldsize))
        loss_quorum = torch.tensor([1]).cuda()

        enableddp = True

    starttime = time.time()
    if not args.noprogress:
        progressprof = Progress()

    batchsize = train_params.batchsize
    learning_rate = train_params.init_learning_rate

    # build dataset & testing dataset
    starttime = time.time()

    base_dir = Path(train_params.base_dir)
    datatype = train_params.dataset
    train_dirs = None
    train_captures = None

    # Load
    if datatype == "ava":
        train_captures, train_dirs = train_csv_loader(base_dir, train_params.data_csv, train_params.nids)
        dataset = AvaMultiCaptureDataset(train_captures, train_dirs, downsample=train_params.downsample)
    else:
        raise ValueError(f"Unsupported dataset: {datatype}")

    vertmean, vertstd = torch.Tensor(dataset.vertmean).cuda(), dataset.vertstd

    maxiter = train_params.maxiter  # that ought to be enough for anyone
    logging.info(" maxiter :  {}, batchsize: {}".format(maxiter, batchsize))

    # dummysampler = AirstoreDummySampler(dataset)
    dummysampler = torch.utils.data.RandomSampler(
        dataset, replacement=True, num_samples=maxiter * batchsize * 2
    )  # infinite random sampler

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        sampler=dummysampler,
        shuffle=False,
        drop_last=True,
        num_workers=numworkers,
        collate_fn=none_collate_fn,
    )

    # Store neut avgtex and neut vert from all ids for x-id check
    all_neut_avgtex_vert = []

    for directory in train_dirs:
        _, neut_avgtex, neut_vert = get_framelist_neuttex_and_neutvert(directory)

        neut_avgtex = (neut_avgtex - dataset.texmean) / dataset.texstd
        neut_verts = (neut_vert - dataset.vertmean) / dataset.vertstd
        all_neut_avgtex_vert.append(
            {
                "neut_avgtex": torch.Tensor(neut_avgtex),
                "neut_verts": torch.Tensor(neut_verts),
            }
        )

    # Driver data for x-id
    driver_dataset = AvaMultiCaptureDataset(
        train_captures,
        train_dirs,
        downsample=train_params.downsample,
        cameras_specified=FRONTAL_CAMERAS,
    )

    driver_dataloader = torch.utils.data.DataLoader(
        driver_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        collate_fn=none_collate_fn,
    )

    logging.info("Datasets instantiated ({:.2f} s)".format(time.time() - starttime))

    # data writer
    starttime = time.time()
    if not args.noprogress:
        writer = progressprof.get_writer()
        logging.debug("Writer instantiated ({:.2f} s)".format(time.time() - starttime))

    # build autoencoder
    starttime = time.time()
    assetpath = pathlib.Path(__file__).parent / "assets"
    ae = get_autoencoder(dataset, assetpath=str(assetpath))

    # load checkpoint
    if train_params.checkpoint:
        ae = load_checkpoint(ae, train_params.checkpoint)
    ae = ae.to("cuda").train()

    iternum = 0

    if enableddp:
        # TODO(julieta) control whether we want to distribute the full model, or just a subset
        # ae = torch.nn.parallel.DistributedDataParallel(ae, device_ids=[0], find_unused_parameters=False)
        ae = torch.nn.parallel.DistributedDataParallel(ae, device_ids=[0], find_unused_parameters=True)

    optim_type = "adam"
    _, optim = gen_optimizer(
        ae,
        optim_type,
        batchsize,
        args.rank,
        learning_rate,
        tensorboard_logger,
        args.worldsize,
    )
    scheduler = lr_scheduler.StepLR(optim, step_size=train_params.lr_scheduler_iter, gamma=4 / 3)

    logging.info("Autoencoder instantiated ({:.2f} s)".format(time.time() - starttime))

    starttime = time.time()

    loss_weights: Dict[str, float] = {
        "irgbl1": train_params.losses.irgbl1,
        "vertl1": train_params.losses.vertl1,
        "kldiv": train_params.losses.kldiv,
        "primvolsum": train_params.losses.primvolsum,
    }

    logging.info("Optimizer instantiated ({:.2f} s)".format(time.time() - starttime))

    # train
    starttime = time.time()
    prevloss = np.inf

    # Total experiment time
    lstart = time.time()

    prevtime = time.time()

    output_set = set(train_params.output_set)
    logging.info("OUTPUT SET :{}".format(output_set))

    driver_dataiter = iter(driver_dataloader)

    iternum = 0
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

        output = ae(
            cudadata["camrot"],
            cudadata["campos"],
            cudadata["focal"],
            cudadata["princpt"],
            cudadata["modelmatrix"],
            cudadata["avgtex"],
            cudadata["verts"],
            cudadata["neut_avgtex"],
            cudadata["neut_verts"],
            cudadata["pixelcoords"],
            cudadata["idindex"],
            cudadata["camindex"],
            running_avg_scale=running_avg_scale,
            # These control the behaviour of the forward pass, and make the optimization easier/harder and more/less stable
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

        # TODO(julieta) DECIDE: do we want to keep explosion detection? Probably not?
        # local_explosion = not args.nostab and (loss.item() > 400 * prevloss or not np.isfinite(loss.item()))
        # loss_explosion = False
        # if enableddp:
        #     loss_quorum[0] = 1
        #     if local_explosion:
        #         loss_quorum[0] = 0
        #         logging.warning(f"rank {args.rank} is seeing unstable loss value(s): {loss.item()}")
        #     loss_explosion = check_quorum(loss_quorum, allmembers)
        # else:
        #     if local_explosion:
        #         logging.warning(f"rank {args.rank} is seeing unstable loss value(s): {loss.item()}")
        #         loss_explosion = True

        # if local_explosion:
        #     # raise RuntimeError("Cannot recover from loss explosion!")
        #     try:
        #         logging.warning(
        #             f"Rank {args.rank} found a sample with loss explosion for id {cudadata['idindex'].tolist()}, setting no grad"
        #         )
        #     except Exception as e:
        #         print(e)
        #     # loss.register_hook(lambda grad: torch.zeros_like(grad))
        #     # loss_explosion = False

        optim.zero_grad()
        loss.backward()
        params = [p for pg in optim.param_groups for p in pg["params"]]

        for p in params:
            if hasattr(p, "grad") and p.grad is not None:
                p.grad.data[torch.isnan(p.grad.data)] = 0
                p.grad.data[torch.isinf(p.grad.data)] = 0

        torch.nn.utils.clip_grad_norm_(ae.parameters(), train_params.clip)
        optim.step()

        # Compute iter total time anyway -- no extra syncing needed, there is already an implicity sync during `backward`
        # and an explicit one to check for loss explosion
        iter_total_time = time.time() - iter_start_time

        imgout = None
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

            if args.rank == 0:
                imgout = render_img(renderImages, f"{outpath}/progress_{iternum}.png")

            # cross id generation
            if config.progress.cross_id:

                indices_subjects = random.sample(
                    range(0, len(all_neut_avgtex_vert)),
                    config.progress.cross_id_n_subjects,
                )
                indices_subjects.sort()

                driver = next(driver_dataiter)
                while driver is None:
                    driver = next(driver_dataiter)

                cudadriver: Dict[str, Union[torch.Tensor, int, str]] = tocuda(driver)

                gt = cudadriver["image"].detach().cpu().numpy()
                gt = einops.rearrange(gt, "1 c h w -> h w c")
                renderImages_xid = []
                renderImages_xid.append(gt)

                ae.eval()
                running_avg_scale = False
                gt_geo = None
                residuals_weight = 1.0

                output_driver = ae(
                    cudadriver["camrot"],
                    cudadriver["campos"],
                    cudadriver["focal"],
                    cudadriver["princpt"],
                    cudadriver["modelmatrix"],
                    cudadriver["avgtex"],
                    cudadriver["verts"],
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

                    output_driven = ae(
                        cudadriver["camrot"],
                        cudadriver["campos"],
                        cudadriver["focal"],
                        cudadriver["princpt"],
                        cudadriver["modelmatrix"],
                        cudadriver["avgtex"],
                        cudadriver["verts"],
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
                ae.train()

                del cudadriver

                if args.rank == 0:
                    imgout = render_img([renderImages_xid], f"{outpath}/x-id/progress_{iternum}.png")

        save_checkpoints = False
        if iternum < 10_000:
            # to account for early loss explosions
            if iternum % 2_000 == 0:
                save_checkpoints = True
        else:
            if iternum % 20_000 == 0:
                save_checkpoints = True

        if save_checkpoints:
            logging.warning(f"rank {args.rank} save checkpoint to outpath {outpath}")
            torch.save(ae.state_dict(), "{}/aeparams.pt".format(outpath))
            torch.save(ae.state_dict(), "{}/aeparams_{:06d}.pt".format(outpath, iternum))

        del cudadata

        if not args.nodisplayloss:
            logging.info(
                "Rank {} Iteration {} loss = {:.4f}, ".format(args.rank, iternum, float(loss.item()))
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

        if iternum >= maxiter:
            logging.info(
                f"Stopping training due to max iter limit, rank {args.rank} curr iter {iternum} max allowed iter {maxiter}"
            )
            lend = time.time()
            totaltime = lend - lstart
            times = {"totaltime": totaltime, "maxiter": iternum}
            np.save(f"{outpath}/timesinfo_r{args.rank}", times, allow_pickle=True)

            logging.info(
                "Rank {} Iteration {} loss = {:.5f}, ".format(args.rank, iternum, float(loss.item()))
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
    writer.finalize()
