"""Dump latent variables from the model"""

import os
import sys
import pathlib
import pandas as pd
import torch
import importlib
import importlib.util
from data.nr_dataset import MultiCaptureDataset, MugsyCapture

os.environ["RSC_AVATAR_PYUTILS_PATH"] = "/checkpoint/avatar/jinkyuk/pyutils"
sys.path.append(os.getenv('RSC_AVATAR_PYUTILS_PATH'))
import pyutils

def import_module(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    pass


if __name__ == "__main__":
    # main()

    # Load 200-model dataset
    downsample = 4
    captures = pd.read_csv(pathlib.Path(__file__).parent / "217_ids.csv", dtype=str)
    captures = [MugsyCapture(mcd=row['mcd'], mct=row['mct'], sid=row['sid']) for _, row in captures.iterrows()]
    train_captures = captures[:200]
    dataset = MultiCaptureDataset(train_captures, downsample=downsample)

    # Load model
    experconfig = import_module(pathlib.Path(__file__).parent / "config.py", "config")
    # unparsed_args = {k: v for k, v in vars(args).items() if k not in parsed}
    # logging.info(f"Unparsed args: {unparsed_args}")
    profile = getattr(experconfig, "Train")# (**unparsed_args)
    ae = profile.get_autoencoder(None, dataset)

    checkpoint = torch.load("/checkpoint/avatar/julietamartinez/oss_release/run-8-nodes-64-gpus-1-dl-workers/2023-10-29T03_39_05.587606/0/0/aeparams_080000.pt")
    remove_prefix = 'module.'
    checkpoint = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in checkpoint.items()}
    ae.load_state_dict(checkpoint, strict=True)
