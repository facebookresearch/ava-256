import numpy as np
import pandas as pd
import pytest

from data.ava_dataset import getitem
from utils import load_krt


def test_getitem_avaSingleCaptureDataset():
    """Ensure that __getitem__ is doing things correctly for AVA SingleCaptureDataset"""
    num_frames = 10

    framelist = pd.DataFrame.from_dict(
        {"seg_id": ["E001_Neutral_Eyes_Open" for _ in range(num_frames)], "frame_id": [j for j in range(num_frames)]}
    )

    krt_file = "assets/KRT"
    krt_dicts = load_krt(krt_file)
    cameras = list(krt_dicts.keys())

    for f in range(num_frames):
        for c, camera in enumerate(cameras):
            item = __getitem__(f * len(cameras) + c, framelist, cameras)
            assert item[0] == "E001_Neutral_Eyes_Open"
            assert item[1] == f
            assert item[2] == camera
