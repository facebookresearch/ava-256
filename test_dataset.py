import os
import sys
import torch
import logging
import einops
import pathlib
from PIL import Image
import pickle

import pandas as pd
import numpy as np
from data.nr_dataset import SingleCaptureDataset, MultiCaptureDataset, MugsyCapture
# from data.mgr_dataset import SingleCaptureDataset

from joblib import Parallel, delayed


root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

# ava sync neurvol2-jason; SCENV=ava rsc_launcher launch \
#   --projects AIRSTORE_AVATAR_RSC_DATA_PIPELINE_CRYPTO \
#   -e 'source /uca/conda-envs/activate-latest && cd /home/$USER/rsc/neurvol2-jason/ && python3 test_dataset.py'

# def worker_init_fn_air(workerid):
#     print(f"{workerid=}")

def create_single_capture_dataset(capture: MugsyCapture):
    return SingleCaptureDataset(capture, downsample=4)

def test_mugsy():
    # capture = MugsyCapture("20210223", "1023", "avw368")
    captures = pd.read_csv(pathlib.Path(__file__).parent / "217_ids.csv", dtype=str)
    captures = [MugsyCapture(mcd=row['mcd'], mct=row['mct'], sid=row['sid']) for _, row in captures.iterrows()]

    # train_captures = captures[:200]
    # captures = captures[::-1]
    # dataset = MultiCaptureDataset(captures)
    # dataset = SingleCaptureDataset(capture)

    parallel_pool = Parallel(n_jobs=8, verbose=1)
    delayed_funcs = []
    for capture in captures:
        cache_dir = f"/uca/julieta/cache/m--{capture.mcd}--{capture.mct}--{capture.sid.upper()}--GHS/cambyte_transforms.pkl"
        if os.path.exists(cache_dir):
            continue
        delayed_funcs.append(delayed(create_single_capture_dataset)(capture))

    # for capture in captures:
        # self.single_capture_datasets[capture] = SingleCaptureDataset(capture, downsample)

    parallel_pool(delayed_funcs)
    quit()

    # dummysampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=1000000000) # infinite random sampler
    batchsize = 1
    numworker = 0
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batchsize,
                                             shuffle=False,
                                             drop_last=True,
                                             num_workers=numworker,
                                             pin_memory=True,
                                            )

    for i, d in enumerate(dataloader):
    # d = dataaloader[("057726", "401071")]
    # for i, d in enumerate(dataset):

        print()
        for k, v in d.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                print(k, v.shape)
            else:
                print(k, type(v))

            if k == "image":
                img = einops.rearrange(v, "1 c h w -> h w c")
                im = Image.fromarray(img.numpy().astype(np.uint8))
                im.save("/checkpoint/avatar/julietamartinez/oss_release/meow_new.png")

        if i >= 0:
            with open('/checkpoint/avatar/julietamartinez/oss_release/new.pkl', 'wb') as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

            break

def test_mgr():
    dataset = SingleCaptureDataset("999741924309556")

if __name__ == "__main__":
    test_mugsy()
    # test_mgr()
