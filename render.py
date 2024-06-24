# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
from typing import Dict, Union

import einops
import torch
import yaml
from tqdm import tqdm

from data.ava_dataset import MultiCaptureDataset as AvaMultiCaptureDataset
from data.ava_dataset import SingleCaptureDataset as AvaSingleCaptureDataset
from data.ava_dataset import none_collate_fn
from data.utils import MugsyCapture, get_framelist_neuttex_and_neutvert
from fvcore.common.config import CfgNode as CN
from utils import get_autoencoder, load_checkpoint, render_img, tocuda, train_csv_loader, xid_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Cross ID driving")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/aeparams_300000.pt", help="checkpoint location")
    parser.add_argument("--output-dir", type=str, default="viz/", help="output image directory")
    parser.add_argument("--config", default="configs/config-4.yaml", type=str, help="config yaml file")

    # Cross ID visualization configuration
    parser.add_argument("--driver-id", type=str, default="20230405--1635--AAN112", help="id of the driver avatar")
    parser.add_argument("--driven-id-indices", type=list, default=[1,2,3], help="id of the driven avatar")
    parser.add_argument("--camera-id", type=str, default="401031", help="render camera id")
    parser.add_argument(
        "--segment-id",
        type=str,
        default="EXP_jaw005",
        help="segment to render; render all available frames if None",
    )
    parser.add_argument("--opts", default=[], type=str, nargs="+")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = CN(yaml.load(file, Loader=yaml.UnsafeLoader))

    config.merge_from_list(args.opts)

    train_params = config.train

    output_dir = args.output_dir + "/" + args.driver_id + "+" + args.segment_id

    pathlib.Path(output_dir + "/x-id").mkdir(parents=True, exist_ok=True)

    # Train dataset mean/std texture and vertex for normalization
    train_captures, train_dirs = train_csv_loader(train_params.dataset_dir, train_params.data_csv, train_params.nids)
    dataset = AvaMultiCaptureDataset(train_captures, train_dirs, downsample=train_params.downsample)

    batchsize = 1
    numworkers = 1

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        drop_last=True,
        num_workers=numworkers,
        collate_fn=none_collate_fn,
    )

    # Get Autoencoder
    assetpath = pathlib.Path(__file__).parent / "assets"
    ae = get_autoencoder(dataset, assetpath=assetpath)

    # Load from checkpoint
    ae = load_checkpoint(ae, args.checkpoint).cuda()

    # Set to Evaluation mode
    ae.eval()

    # Driver capture dataloader
    driver_capture = MugsyCapture(
        mcd=args.driver_id.split("--")[0], mct=args.driver_id.split("--")[1], sid=args.driver_id.split("--")[2]
    )
    driver_dir = f"{train_params.dataset_dir}/{args.driver_id}/decoder"
    driver_dataset = AvaSingleCaptureDataset(driver_capture, driver_dir, downsample=train_params.downsample)

    texmean = dataset.texmean
    vertmean = dataset.vertmean
    texstd = dataset.texstd
    vertstd = dataset.vertstd

    # Delete dataset because it is no longer used
    del dataset

    # Grab driven normalization stats
    for dataset in [driver_dataset]: #,driven_dataset]:
        dataset.texmean = texmean
        dataset.texstd = texstd
        dataset.vertmean = vertmean
        dataset.vertstd = vertstd

    # select only desired segments
    if args.segment_id:
        driver_dataset.framelist = driver_dataset.framelist.loc[driver_dataset.framelist["seg_id"] == args.segment_id]
    if driver_dataset.framelist.values.tolist() == []:
        raise ValueError(
            f"Asked to render Segment {args.segment_id}, but there are no frames with that Segment in {driver_capture}"
        )

    # select only desired cameras
    if args.camera_id:
        if args.camera_id in driver_dataset.cameras:
            driver_dataset.cameras = [args.camera_id]
        else:
            raise ValueError(f"Camera id {args.camera_id} is not defined for {driver_capture}")
    else:
        # TODO (Emily): Generalize choosing from frontal cameras i.e. ["401031", "401880", "401878"]
        driver_dataset.cameras = ["401031"]

    driver_loader = torch.utils.data.DataLoader(
        driver_dataset,
        batch_size=batchsize,
        shuffle=False,
        drop_last=False,
        num_workers=numworkers,
        collate_fn=none_collate_fn,
    )
    
    driver_dataiter = iter(driver_loader)

    it = 0
    
    # Store neut avgtex and neut vert from all ids for x-id check
    all_neut_avgtex_vert = []

    for directory in train_dirs:
        _, neut_avgtex, neut_vert = get_framelist_neuttex_and_neutvert(directory)

        neut_avgtex = (neut_avgtex - dataset.texmean) / dataset.texstd
        neut_verts = (neut_vert - dataset.vertmean) / dataset.vertstd
        all_neut_avgtex_vert.append({"neut_avgtex": torch.tensor(neut_avgtex), "neut_verts": torch.tensor(neut_verts)})
    
    output_set = set(train_params.output_set)
    it = 0

    while driver_dataiter:
        xid_eval(ae, driver_dataiter, all_neut_avgtex_vert, config, output_set, output_dir, 0, it, indices_subjects=args.driven_id_indices)
        it += 1
        
    
    print(f"Done! Saved {it} images to {output_dir}")
