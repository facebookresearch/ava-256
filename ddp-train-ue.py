# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Train a (really simple) universal facial encoder."""

import argparse
import logging
import platform
import os
import copy
import sys
import time
from typing import Dict, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from importlib import import_module
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from data.headset_dataset import MultiCaptureDataset as HeadsetMultiCaptureDataset
from data.ava_dataset import none_collate_fn
from fvcore.common.config import CfgNode as CN
from models.headset_encoders.universal import UniversalEncoder
from models.headset_encoders.loss import UniversalEncoderLoss
from models.headset_encoders.ud import UDWrapper
from utils import load_checkpoint, tocuda, train_headset_csv_loader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
sys.dont_write_bytecode = True
ddp_train = import_module("ddp-train")


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
formatter = logging.Formatter("%(asctime)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

torch.backends.cudnn.benchmark = True  # gotta go fast!


def cleanup():
    dist.destroy_process_group()


def prepare(rank, world_size, train_params):
    """
    Load dataset from the specified asset directories and CSV files
    """
    base_dir = train_params.dataset_dir
    latent_code_dir = train_params.gt_dir
    train_dirs = None
    train_captures = None

    # Decoder avatar identities
    df = pd.read_csv(train_params.data_csv, dtype=str)
    dec_ids = [f"{capture.sid}_{capture.mcd}--0000" for capture in df.itertuples()]
    enc_ids = [f"{capture.sid}_{capture.hcd}--{capture.hct}" for capture in df.itertuples()]
    enc_ids_train = enc_ids[:max(int(len(enc_ids) * 0.15), 1)]
    enc_ids_valid = enc_ids[max(int(len(enc_ids) * 0.15), 1):]

    # Load dataset, should come from 256_ids_enc.csv
    train_captures, train_dirs = train_headset_csv_loader(base_dir, train_params.data_csv, identities=enc_ids_train)
    _, train_gt_dirs = train_headset_csv_loader(latent_code_dir, train_params.data_csv, identities=enc_ids_train)
    train_dataset = HeadsetMultiCaptureDataset(
        train_captures,
        train_dirs,
        train_gt_dirs,
        downsample=2.083
    )

    valid_captures, valid_dirs = train_headset_csv_loader(base_dir, train_params.data_csv, identities=enc_ids_valid)
    _, valid_gt_dirs = train_headset_csv_loader(latent_code_dir, train_params.data_csv, identities=enc_ids_valid)
    valid_dataset = HeadsetMultiCaptureDataset(
        valid_captures,
        valid_dirs,
        valid_gt_dirs,
        downsample=2.083
    )
    datasets = {"train": train_dataset, "valid": valid_dataset}

    maxiter = train_params.maxiter
    if rank == 0:
        logging.info(f" Train maxiter : {maxiter}, batchsize: {train_params.batchsize}")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_params.batchsize,
        num_workers=train_params.num_workers,
        pin_memory=False,
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=none_collate_fn,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=train_params.batchsize,
        num_workers=train_params.num_workers,
        pin_memory=False,
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=none_collate_fn,
    )
    dataloaders = {"train": train_dataloader, "valid": valid_dataloader}
    return datasets, dataloaders, dec_ids, enc_ids


def print_loss(log):
    log = copy.deepcopy(log)
    for k in log:
        log[k] = np.mean(log[k])
    log_loss_txt = "Losses: " + ", ".join(["{}={:.4f}".format(k, v.mean()) for k, v in log.items()])
    return log_loss_txt


def warmup_dataloader(loader):
    """Warms up the dataloader by iterating over it once"""
    for i, data in enumerate(loader):
        if i > 2:
            break


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

    def selective_logging_info(msg: str):
        if rank == 0:
            logging.info(msg)

    logging.info(f"{rank=}, {args.world_size=}")
    torch.manual_seed(rank)
    np.random.seed(rank)
    ddp_train.setup(rank, args.world_rank, args.world_size, args.masteraddress, args.masterport)

    train_params = config.train
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    outpath = os.path.join(current_dir, config.progress.output_path)
    tensorboard_logger = None
    if config.progress.tensorboard.logdir is not None and args.world_rank == 0:
        tensorboard_logdir = config.progress.output_path + "/" + config.progress.tensorboard.logdir
        logging.info(f"Creating tensorboard output at {tensorboard_logdir}")
        tensorboard_logger = SummaryWriter(tensorboard_logdir)

    datasets, dataloaders, dec_ids, _ = prepare(args.world_rank, args.world_size, train_params)

    # 1. Create model, optimizer, scheduler
    starttime = time.time()
    ue = UniversalEncoder(in_chans=1, out_chans=256, num_views=4)
    ud = UDWrapper(ud_exp_name="/uca/leochli/oss/ava256_universal_decoder", identities=dec_ids)
    model = UniversalEncoderLoss(ue, ud, identities=dec_ids).to(rank)
    if train_params.checkpoint:
        selective_logging_info(f"Loading checkpoint from {train_params.checkpoint}")
        model = load_checkpoint(model, train_params.checkpoint)
    if args.world_size > 1:
        selective_logging_info("Converting BN to SyncBN...")
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    para_model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    initial_lr = train_params.init_learning_rate
    _, optim = ddp_train.gen_optimizer(
        model,
        "adam",
        train_params.batchsize,
        rank=rank,
        learning_rate=initial_lr,
        tensorboard_logger=tensorboard_logger,
        worldsize=args.worldsize,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=train_params.maxiter)
    if train_params.scheduler_checkpoint:
        selective_logging_info(f"Loading scheduler checkpoint from {train_params.scheduler_checkpoint}")
        scheduler.load_state_dict(train_params.scheduler_checkpoint)

    selective_logging_info("Model/optimizer/scheduler instantiated ({:.2f} s)".format(time.time() - starttime))

    # 2. Warmup dataloader and prefetching
    starttime = time.time()
    selective_logging_info("Warming up dataloader and prefetching...")
    warmup_dataloader(dataloaders["train"])
    warmup_dataloader(dataloaders["valid"])
    selective_logging_info("Done warming up and prefetching ({:.2f} s)".format(time.time() - starttime))

    # 3. Train
    starttime = time.time()
    loss_weights: Dict[str, float] = {
        "img_L1": train_params.losses.img_L1,
        "expression": train_params.losses.expression,
        "geo_weighted_L1": train_params.losses.geo_weighted_L1
    }
    iternum = scheduler.last_epoch
    iter_total_time = 0
    if args.world_rank == 0:
        train_loss_log = defaultdict(list)  # Keep a loss log iff we're on rank 0
    for data in dataloaders["train"]:
        if iternum >= train_params.maxiter:
            break
        iternum += 1
        # data should contain {"headset_cam_imgs", "cond_headset_cam_img", "gt_latent_code", "index"}
        if data is None:
            continue
        iter_start_time = time.time()
        
        # TODO: Add data augmentation here.
        cudadata: Dict[str, Union[torch.Tensor, int, str]] = tocuda(data)
        optim.zero_grad()
        output, losses = para_model(cudadata, loss_weights)

        loss = sum(loss_weights[k] * v.mean() for k, v in losses.items())
        loss.backward()
        optim.step()
        scheduler.step()

        iter_total_time += time.time() - iter_start_time

        if args.world_rank == 0 and iternum > 0:
            for k, v in losses.items():
                train_loss_log[k].append(v.mean().item())
            train_loss_log["loss"].append(loss.item())
            if iternum % train_params.log_freq == 0:
                log_loss_txt = print_loss(train_loss_log)
                logging.info(f"Iter {iternum} | {log_loss_txt} | time: {iter_total_time / train_params.log_freq:.3f} s")
                train_loss_log = defaultdict(list)
                iter_total_time = 0
            if iternum % train_params.viz_freq == 0:
                viz = model.get_visual(output)
                rgb_viz = Image.fromarray(np.clip(viz, 0, 255).astype(np.uint8))
                rgb_viz.save(f"{outpath}/train_progress_{iternum}.png")
            if iternum % train_params.eval_freq == 0:
                model.eval()
                valid_loss_log = defaultdict(list)
                with torch.no_grad():
                    for i, val_data in enumerate(dataloaders["valid"]):
                        val_cudadata = tocuda(val_data)
                        val_output, val_losses = model(val_cudadata, loss_weights)
                        val_loss = sum(loss_weights[k] * v.mean() for k, v in losses.items())
                        for k, v in val_losses.items():
                            valid_loss_log[k].append(v.mean().item())
                        valid_loss_log["loss"].append(val_loss.item())
                        if i > 10:
                            break
                    val_log_loss_txt = print_loss(valid_loss_log)
                    logging.info(f"Validation @ Iter {iternum} | {val_log_loss_txt}")
                    val_output, _ = model(val_cudadata, loss_weights, use_face_mask=False)
                    val_rgb_viz = model.get_visual(val_output)
                    val_rgb_viz = Image.fromarray(np.clip(val_rgb_viz, 0, 255).astype(np.uint8))
                    val_rgb_viz.save(f"{outpath}/valid_{iternum}.png")
                model.train()

                # Save checkpoints after each validation
                torch.save(scheduler.state_dict(), f"{outpath}/scheduler_{iternum}.pt")
                torch.save(model.state_dict(), f"{outpath}/encoder_{iternum}.pt")

            if tensorboard_logger is not None and iternum % config.progress.tensorboard.log_freq == 0:
                tensorboard_logger.add_scalar("Total Loss", float(loss.item()), iternum)
                for k, v in losses.items():
                    tensorboard_logger.add_scalar(k, float(v.mean().item()), iternum)

        del cudadata
    cleanup()


if __name__ == "__main__":
    __spec__ = None  # to use ipdb

    # parse arguments
    parser = argparse.ArgumentParser(description="Train an autoencoder")

    # Can overwrite some of these parameter to, eg, train over multiple hosts
    parser.add_argument("--worldsize", type=int, default=1, help="the number of processes for distributed training")
    parser.add_argument("--masteraddress", type=str, default="localhost", help="master node address (hostname or ip)")
    parser.add_argument("--masterport", type=str, default="43322", help="master node network port")

    parser.add_argument("--config", default="configs/config_universal_encoder.yaml", type=str, help="config yaml file")
    parser.add_argument("--opts", default=[], type=str, nargs="+")

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = CN(yaml.load(file, Loader=yaml.UnsafeLoader))

    config.merge_from_list(args.opts)

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
