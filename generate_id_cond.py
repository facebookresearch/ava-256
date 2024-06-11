# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
from typing import Dict, Union

import torch
import yaml
from fvcore.common.config import CfgNode as CN
from tqdm import tqdm

from data.ava_dataset import MultiCaptureDataset as AvaMultiCaptureDataset
from utils import get_autoencoder, load_checkpoint, tocuda, train_csv_loader

if __name__ == "__main__":
    """
    This is a script to generate and dump the id conditioning outputs for all identities. For downstream
    tasks where the universal decoder is already trained, it would be easier to use the cached identity conditioning outputs.
    """
    parser = argparse.ArgumentParser(
        description="Generate id conditioning outputs for all identities"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/aeparams.pt",
        help="checkpoint location",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="identity_conditioning/",
        help="output directory",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml", type=str, help="config yaml file"
    )

    parser.add_argument("--opts", default=[], type=str, nargs="+")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = CN(yaml.load(file, Loader=yaml.UnsafeLoader))

    config.merge_from_list(args.opts)

    train_params = config.train

    output_dir = args.output_dir + "/" + "identity_conditioning"

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Train dataset mean/std texture and vertex for normalization
    train_captures, train_dirs = train_csv_loader(
        train_params.dataset_dir, train_params.data_csv, train_params.nids
    )
    dataset = AvaMultiCaptureDataset(
        train_captures, train_dirs, downsample=train_params.downsample
    )

    # Serialize and pickle the dataset
    torch.save(dataset, output_dir + "/ae_info.pkl")

    # Get Autoencoder
    assetpath = pathlib.Path(__file__).parent / "assets"
    ae = get_autoencoder(dataset, assetpath=assetpath)

    # Load from checkpoint
    ae = load_checkpoint(ae, args.checkpoint).cuda()

    # Set to Evaluation mode
    ae.eval()

    texmean = dataset.texmean
    vertmean = dataset.vertmean
    texstd = dataset.texstd
    vertstd = dataset.vertstd

    for single_capture, single_dataset in tqdm(
        dataset.single_capture_datasets.items(), desc="Captures"
    ):
        # Get neutral
        neut_avgtex, neut_vert = single_dataset.neut_avgtex, single_dataset.neut_vert
        # Normalize
        neut_avgtex = (neut_avgtex - texmean) / texstd
        neut_verts = (neut_vert - vertmean) / vertstd
        data = {
            "neut_avgtex": torch.from_numpy(neut_avgtex[None]),
            "neut_verts": torch.from_numpy(neut_verts[None]),
        }

        cuda_data: Dict[str, Union[torch.Tensor, int, str]] = tocuda(data)

        # Generate id_cond from the capture
        # {z_tex, b_tex, z_geo, b_geo}
        id_cond = ae.id_encoder(cuda_data["neut_verts"], cuda_data["neut_avgtex"])

        ident_str = f"{single_capture.sid}_{single_capture.mcd}--0000"
        save_dir = output_dir + "/" + ident_str

        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        torch.save(id_cond, f"{save_dir}/id_cond.pkl")

    print(
        f"Done! Dumped {len(dataset.single_capture_datasets)} id_cond to {output_dir}"
    )
