# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Train an autoencoder."""

import itertools
import logging
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from collections import defaultdict
import einops
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from omegaconf import DictConfig, OmegaConf
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

# FRONTAL_CAMERAS = ["401168", "401875", "402040", "401031"]

# TODO(julieta) see if this actually does anything
torch.backends.cudnn.benchmark = True  # gotta go fast!


def prepare(
    train_params,
    cameras: Optional[List[str]] = None,
):
    base_dir = train_params.dataset_dir
    train_dirs = None
    train_captures = None

    # Load dataset
    train_captures, train_dirs = train_csv_loader(base_dir, train_params.data_csv, train_params.nids)
    
    # FIXME(julieta) this is a hack to get the first 4 captures for testing
    # train_captures = train_captures[:4]
    # train_dirs = train_dirs[:4]

    dataset = AvaMultiCaptureDataset(
        train_captures, 
        train_dirs, 
        downsample=train_params.downsample,
        cameras_specified=cameras,
    )

    starttime = time.time()

    logging.info("batchsize: {}".format(train_params.batchsize))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_params.batchsize,
        sampler=None,
        shuffle=True,
        drop_last=True,
        num_workers=train_params.num_workers,
        collate_fn=none_collate_fn,
    )

    return dataset, dataloader


def test(config):
    """
    Run validation on a smaller set of novel subjects
    """

    train_params, test_params = config.train, config.test
    tensorboard_logger = None

    # Create train dataset
    train_dataset, _ = prepare(train_params)
    
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    outpath = os.path.join(current_dir, config.progress.output_path)
    os.makedirs(f"{outpath}/eval_viz", exist_ok=True)

    # Create model
    starttime = time.time()
    assetpath = Path(__file__).parent / "assets"
    ae = get_autoencoder(train_dataset, assetpath=str(assetpath))

    if not test_params.checkpoint:
        raise ValueError("No checkpoint given for testing")
    
    # Load model checkpoint
    logging.info(f"Loading checkpoint from {test_params.checkpoint}")
    ae = load_checkpoint(ae, test_params.checkpoint)
    numbers = re.findall(r"\d+", test_params.checkpoint)
    if not numbers:
        raise ValueError(f"Checkpoint given but it does not contain an iteration number: {test_params.checkpoint}")
    iternum = int(numbers[-1])

    # Set the model to val
    model = ae.eval().to("cuda")
    
    # TODO(julieta) add psnr to the losses
    loss_weights: Dict[str, float] = {
        "irgbl1": test_params.losses.irgbl1,
        "vertl1": test_params.losses.vertl1,
        "psnr": test_params.losses.psnr,
    }

    # Create val dataset    
    val_dataset, _ = prepare(test_params)

    # Get norm stats from train dataset
    vertmean, vertstd = torch.Tensor(train_dataset.vertmean).to("cuda"), train_dataset.vertstd

    output_set = set(test_params.output_set)
    logging.info("OUTPUT SET :{}".format(output_set))

    for capture, dataset in val_dataset.single_capture_datasets.items():        
        dataset.vertmean = train_dataset.vertmean
        dataset.vertstd = train_dataset.vertstd
        dataset.texmean = train_dataset.texmean
        dataset.texstd = train_dataset.texstd

        os.makedirs(f"{outpath}/eval_viz/{capture.sid}", exist_ok=True)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=test_params.batchsize,
            sampler=None,
            shuffle=True,
            num_workers=test_params.num_workers,
            collate_fn=none_collate_fn,
        )

        loss_means = defaultdict()
        
        for i, data in enumerate(dataloader):

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

            with torch.no_grad():    
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

            # Add to mean dict
            for name, value in losses.items():
                [name].append(value)

            # compute final loss
            loss = sum(
                [
                    loss_weights[k]
                    * (torch.sum(v[0]) / torch.sum(v[1]).clamp(min=1e-6) if isinstance(v, tuple) else torch.mean(v))
                    for k, v in losses.items()
                ]
            )

            # NOTE(julieta) We compute iter total time anyway -- no extra syncing needed, there is already an implicit
            # sync during `backward`
            iter_total_time = time.time() - iter_start_time

            # print progress
            if (i % 10 == 0):
                renderImages = []
                del output["verts"]
                for b in range(cudadata["camrot"].shape[0]):
                    gt = cudadata["image"][b].detach().cpu().numpy()
                    gt = einops.rearrange(gt, "c h w -> h w c")

                    rgb_orig = output["irgbrec"][b].detach().cpu().numpy()
                    rgb_orig = einops.rearrange(rgb_orig, "c h w -> h w c")
                    renderImages.append([gt, rgb_orig, (gt - rgb_orig) ** 2 * 10])

                    render_img(renderImages, f"{outpath}/eval_viz/{capture.sid}/eval_{i:06d}.png")

            del cudadata

            if not config.progress.nodisplayloss:
                logging.info(
                    "{} iteration {}/{} loss = {:.4f}, ".format(capture.sid, i, len(dataloader), float(loss.item()))
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
            
            if i >= test_params.max_iters:
                
                for name, value in _loss_dict.items():
                    loss_means[name] = np.mean(value)

                print(loss_means)
                print()

                break


if __name__ == "__main__":
    __spec__ = None  # to use ipdb

    config_path: str = sys.argv[1]
    console_commands: List[str] = sys.argv[2:]

    config = OmegaConf.load(config_path)
    config_cli = OmegaConf.from_cli(args_list=console_commands)
    if config_cli:
        logging.info("Overriding with the following args values:")
        logging.info(f"{OmegaConf.to_yaml(config_cli)}")
        config = OmegaConf.merge(config, config_cli)

    logging.info({"Full config:"})
    logging.info(f"{OmegaConf.to_yaml(config)}") 

    # Update worldsize with number of GPUs
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        logging.info(f"Using {ngpus_per_node} GPUs per node.")
    else:
        ngpus_per_node = 1

    test(config)
