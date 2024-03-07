"""Train an autoencoder."""
import argparse
import hashlib as hlib
import importlib
import importlib.util
import itertools
import logging
import os
import pathlib
import sys
import time
from typing import Dict, List, Union
import einops

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from data.ava_dataset import MultiCaptureDataset as AvaMultiCaptureDataset
from data.mini_ava_dataset import MultiCaptureDataset as MiniAvaMultiCaptureDataset
from data.mugsy_dataset import MugsyCapture
from data.mugsy_dataset import MultiCaptureDataset as MugsyMultiCaptureDataset
from data.mugsy_dataset import none_collate_fn
from losses import mean_ell_1
from models.bottlenecks.vae import kl_loss_stable

sys.dont_write_bytecode = True

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

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
    parser = argparse.ArgumentParser(description="Train an autoencoder")
    parser.add_argument("--base-dir", default="/home/ekim2/Storage/MetaProject/datasets/multiface_mini_dataset/", help="base directory for training")
    parser.add_argument("experconfig", type=str, help="experiment config file")
    parser.add_argument("--profile", type=str, default="Train", help="config profile")
    parser.add_argument("--devices", type=int, nargs="+", default=[0], help="devices")
    parser.add_argument("--ablationcamera", action="store_true", help="running ablation camera experiments")
    parser.add_argument("--noprogress", action="store_true", help="don't output training progress images")
    parser.add_argument("--nostab", action="store_true", help="don't check loss stability")
    parser.add_argument("--rank", type=int, default=0, help="process rank in distributed training")
    parser.add_argument("--worldsize", type=int, default=1, help="the number of processes for distributed training")
    parser.add_argument("--masterip", type=str, default="localhost", help="master node ip address")
    parser.add_argument("--masterport", type=str, default="43321", help="master node network port")
    parser.add_argument("--nodisplayloss", action="store_true", help="logging loss value every iteration")
    parser.add_argument("--disableshuffle", action="store_true", help="no shuffle in airstore")
    parser.add_argument("--shard_air", action="store_true", help="no shuffle in airstore")
    parser.add_argument("--log_freq", type=int, default="10", help="tensorboard logging frequency, in training iterations")

    # TODO(julieta) get rid of this, there should only be one dataset in the OSS release
    parser.add_argument("--dataset", type=str, default="mugsy", help="The dataset to use")
    parser.add_argument("--ids2use", type=int, default=-1, help="the number of processes for distributed training")
    parser.add_argument("--idfilepath", type=str, default=None, help="file of id list for training or evaluation")
    parser.add_argument("--tensorboard-logdir", type=str, default=None, help="dir where tensorboard log will output to")

    # Training hyperparams.
    parser.add_argument("--maxiter", type=int, default=10_000_000, help="maximum number of iterations")
    parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--batchsize", type=int, default=2, help="Batch size per GPU to use")
    parser.add_argument("--learning-rate", type=float, default=4e-4, help="Learning rate passed as it to the optimizer")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--nids", type=int, default=1, help="Number of identities to select")
    parser.add_argument("--downsample", type=int, default=6, help="image downsampling factor at data loader")
    args = parser.parse_args()

    # TODO(julieta) remove all references to SLURM variables
    args.worldsize = int(os.environ.get("SLURM_NTASKS", args.worldsize))
    args.rank = int(os.environ.get("SLURM_PROCID", args.rank))

    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    outpath = os.environ.get(
        "RSC_RUN_DIR", os.path.join(current_dir, "run")
    )  # RSC_EXP_RUN_BASE_DIR/SLURM_NODEID/SLURM_LOCALID
    os.makedirs(outpath, exist_ok=True)

    tlogpath = "{}/log-r{}.txt".format(outpath, args.rank)

    # if path exists append automatically
    logging.info(f"Python {sys.version}")
    logging.info(f"PyTorch {torch.__version__}")
    logging.info(" ".join(sys.argv))
    logging.info(f"Output path: {outpath}")

    logging.info(" ==================================================== ")
    logging.info(" args.rank: {}, args: {}".format(args.rank, args))
    v_devices = os.getenv("CUDA_VISIBLE_DEVICES", "NOTFOUND")
    logging.info(" ==================================================== ")
    logging.info(
        " At rank {} DEVICE COUNT : {} -- CUDA VISIBLE DEVICES {}".format(
            args.rank, torch.cuda.device_count(), v_devices
        )
    )
    logging.info(" ==================================================== ")

    cuda_device_count = torch.cuda.device_count()

    tensorboard_logger = None
    if args.tensorboard_logdir is not None and args.rank == 0:
        tensorboard_logdir = args.tensorboard_logdir
        logging.info(f"Creating tensorboard output at {tensorboard_logdir}")
        tensorboard_logger = SummaryWriter(tensorboard_logdir)

    logging.info("@@@@@@@@@@@@@@@ JOB CONFIG:")
    logging.info(args)
    logging.info("@@@@@@@@@@@@@@@ END OF JOB CONFIG \n")

    logging.info("@@@@@@@@@@@@@@@ OS ENV VARIABLES (from sbatch):")
    for k, v in os.environ.items():
        logging.info(f"{k}:{v}")
    logging.info("@@@@@@@@@@@@@@@ END OF OS ENV VARIABLES :")

    logging.info(f"master ip {args.masterip}")

    rank = args.rank
    worldsize = args.worldsize

    disturl = f"tcp://{args.masterip}:{args.masterport}"
    logging.info(f" DIST URL : {disturl}")

    # TODO(julieta) get the number of workers from the command line
    numworkers = args.num_workers

    enableddp = False

    if args.worldsize > 1:
        logging.info(f" INIT DDP RANK {args.rank}  WSIZE {args.worldsize}  URL {disturl}")
        os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
        dist.init_process_group(backend="nccl", init_method=disturl, world_size=args.worldsize, rank=args.rank)
        logging.info(f" distributed training group is initialized at rank : {args.rank}")

        allmembers = None
        loss_quorum = None
        allmembers = dist.new_group(range(worldsize))
        loss_quorum = torch.tensor([1]).cuda()

        enableddp = True

    # load config
    starttime = time.time()
    experconfig = import_module(args.experconfig, "config")
    profile = getattr(experconfig, args.profile)()
    if not args.noprogress:
        progressprof = experconfig.Progress()

    logging.info("@@@@@@@ CONFIG.PY PATH : {}".format(experconfig))
    batchsize = args.batchsize
    learning_rate = args.learning_rate
    logging.info("Config loaded ({:.2f} s)".format(time.time() - starttime))

    # build dataset & testing dataset
    starttime = time.time()

    # Load
    if args.dataset == "mugsy":
        # nr_captures = pd.read_csv(pathlib.Path(__file__).parent / "215_ids.csv", dtype=str)
        # nr_captures = [
        #     MugsyCapture(mcd=row["mcd"], mct=row["mct"], sid=row["sid"], is_relightable=False)
        #     for _, row in nr_captures.iterrows()
        # ]
        r_captures = pd.read_csv(pathlib.Path(__file__).parent / "345_ids.csv", dtype=str)
        r_captures = [
            MugsyCapture(mcd=row["mcd"], mct=row["mct"], sid=row["sid"], is_relightable=True)
            for _, row in r_captures.iterrows()
        ]

        # captures = nr_captures + r_captures
        captures = r_captures

        train_captures = captures[: args.nids]
        # train_captures = np.array_split(train_captures, worldsize)[args.rank]
        # train_captures = captures
        dataset = MugsyMultiCaptureDataset(train_captures, downsample=args.downsample)
    elif args.dataset == "ava":
        # TODO(julieta) do capture objects make sense here? we don't really use them now that we have dirs
        train_captures = [MugsyCapture(mcd="1", mct="1", sid="1")]
        train_dirs = [f"{args.base_dir}/m--20180227--0000--6795937--GHS"]
        dataset = AvaMultiCaptureDataset(train_captures, train_dirs, downsample=args.downsample)
    elif args.dataset == "mini_ava":
        train_captures = [MugsyCapture(mcd="1", mct="1", sid="1")]
        train_dirs = [f"{args.base_dir}/20230405--1635--AAN112"]
        dataset = MiniAvaMultiCaptureDataset(train_captures, train_dirs, downsample=args.downsample)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    vertmean, vertstd = torch.Tensor(dataset.vertmean).cuda(), dataset.vertstd

    maxiter = args.maxiter  # that ought to be enough for anyone
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
        # pin_memory=True,
        collate_fn=none_collate_fn,
    )

    logging.info("Dataset instantiated ({:.2f} s)".format(time.time() - starttime))

    # data writer
    starttime = time.time()
    if not args.noprogress:
        writer = progressprof.get_writer()
        logging.debug("Writer instantiated ({:.2f} s)".format(time.time() - starttime))

    # build autoencoder
    starttime = time.time()
    assetpath = pathlib.Path(__file__).parent / "assets"
    ae = profile.get_autoencoder(dataset, assetpath=str(assetpath))
    ae = ae.to("cuda").train()

    iternum = 0

    if enableddp:
        # TODO(julieta) control whether we want to distribute the full model, or just a subset
        # ae = torch.nn.parallel.DistributedDataParallel(ae, device_ids=[0], find_unused_parameters=False)
        ae = torch.nn.parallel.DistributedDataParallel(ae, device_ids=[0], find_unused_parameters=True)

    optim_type = "adam"
    _, optim = gen_optimizer(ae, optim_type, batchsize, args.rank, learning_rate, tensorboard_logger, args.worldsize)

    logging.info("Autoencoder instantiated ({:.2f} s)".format(time.time() - starttime))

    starttime = time.time()

    loss_weights: Dict[str, float] = profile.get_loss_weights()
    logging.info("Optimizer instantiated ({:.2f} s)".format(time.time() - starttime))

    # train
    starttime = time.time()
    prevloss = np.inf

    # Total experiment time
    lstart = time.time()

    prevtime = time.time()

    output_set = profile.get_output_set()
    logging.info("OUTPUT SET :{}".format(output_set))

    for iternum, data in enumerate(dataloader):
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

        # fmt: off
        # import ipdb; ipdb.set_trace()
        # fmt: on
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

        torch.nn.utils.clip_grad_norm_(ae.parameters(), args.clip)
        optim.step()

        # Compute iter total time anyway -- no extra syncing needed, there is already an implicity sync during `backward`
        # and an explicit one to check for loss explosion
        iter_total_time = time.time() - iter_start_time

        imgout = None
        # # print progress
        if (iternum < 10000 and iternum % 100 == 0) or iternum % 1000 == 0:
            del output["verts"]
            # fmt: off
            # import ipdb; ipdb.set_trace()
            # fmt: on
            if args.rank == 0:
                imgout = writer.batch(
                    iternum,
                    iternum * batchsize + torch.arange(0),
                    f"{outpath}/progress_{iternum}.png",
                    **cudadata,
                    **output,
                )

        # compute evaluation output
        # if not args.noprogress and iternum in evalpoints and args.rank == 0:
        #    writer.batch(iternum, iternum * batchsize + torch.arange(0), **cudadata, **output)

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
        else:
            logging.warning(f"skipping checkpoint save, rank {args.rank} is seeing unstable loss value(s)")

        # if local_explosion:
        # if loss_explosion:
        #     if enableddp:
        #         # to avoid any race conditions when multiple ranks are facing loss explosions
        #         # we do not want to load aeparams.pt while rank 0 is still writing the file
        #         dist.barrier()

        #     base_dir = outpath
        #     if not base_dir:
        #         raise AssertionError("cannot reset without providing base_dir, check env var RSC_EXP_RUN_BASE_DIR")
        #     logging.warning(
        #         f"unstable loss function; resetting -- resume from the latest checkpoint : {outpath}/aeparams.pt"
        #     )
        #     # reloading should called after backward is invoked to clean up CUDA memory of the failed iteration
        #     ae.load_state_dict(torch.load(f"{outpath}/aeparams.pt"), strict=False)
        #     _, optim = gen_optimizer(
        #         ae, optim_type, batchsize, args.rank, learning_rate, tensorboard_logger, encoder_lr=args.encoder_lr
        #     )
        # # else:
        # prevloss = loss.item()

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
        if tensorboard_logger is not None and iternum % args.log_freq == 0:
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

            try:
                tensorboard_logger.add_image("progress", einops.rearrange(imgout, 'h w c -> c h w'), iternum)
            except:
                raise ValueError("Tensorboard cannot log images because it is None.")



        # log losses to tensorboard even if we were not asked for profile stats
        # if tensorboard_logger is not None and iternum % 50 == 0:
        #     tb_loss_stats = {
        #         "rank": args.rank,
        #         "loss": float(loss.item()),
        #     }
        #     for k, v in losses.items():
        #         if isinstance(v, tuple):
        #             tb_loss_stats[k] = float(torch.sum(v[0]) / torch.sum(v[1].clamp(min=1e-6)))
        #         else:
        #             tb_loss_stats[k] = torch.mean(v)

        #     for k, v in tb_loss_stats.items():
        #         tensorboard_logger.add_scalar("loss/" + k, v, iternum)

        #     if enableddp and args.worldsize > 1:
        #         averaged_losses = {k: torch.Tensor([v]).cuda() for k, v in tb_loss_stats.items()}
        #         for k, v in averaged_losses.items():
        #             dist.all_reduce(v, op=dist.ReduceOp.AVG, group=allmembers)
        #             averaged_losses[k] = v

        #         if rank == 0:
        #             for k, v in averaged_losses.items():
        #                 tensorboard_logger.add_scalar(f"averaged_{k}/", v.item())

        # if args.worldsize > 1 and iternum % 50 == 0:
        #     tb_loss_stats = {
        #         "rank": args.rank,
        #         "loss": float(loss.item()),
        #     }
        #     for k, v in losses.items():
        #         if isinstance(v, tuple):
        #             tb_loss_stats[k] = float(torch.sum(v[0]) / torch.sum(v[1].clamp(min=1e-6)))
        #         else:
        #             tb_loss_stats[k] = torch.mean(v)

        #     averaged_losses = {k: torch.Tensor([v]).cuda() for k, v in tb_loss_stats.items()}
        #     for k, v in averaged_losses.items():
        #         dist.all_reduce(v, op=dist.ReduceOp.AVG, group=allmembers)
        #         averaged_losses[k] = v

        #     if rank == 0:
        #         for k, v in averaged_losses.items():
        #             logging.info(f"Iteration {iternum}, averaged {k}: {v.item():.3f}")

        # if iternum and iternum % 2000 == 0:
        #     logging.info(
        #         "Rank {} Iteration {} loss = {:.5f}, ".format(args.rank, iternum, float(loss.item()))
        #         + ", ".join(
        #             [
        #                 "{} = {:.5f}".format(
        #                     k,
        #                     float(
        #                         torch.sum(v[0]) / torch.sum(v[1].clamp(min=1e-6))
        #                         if isinstance(v, tuple)
        #                         else torch.mean(v)
        #                     ),
        #                 )
        #                 for k, v in losses.items()
        #             ]
        #         )
        #     )

        # iternum += 1

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

    # cleanup
    writer.finalize()
