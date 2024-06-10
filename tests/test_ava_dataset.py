# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import pytest

from data.ava_dataset import getitem
from utils import load_camera_calibration


def test_getitem_avaSingleCaptureDataset():
    """Ensure that __getitem__ is doing things correctly for AVA SingleCaptureDataset"""
    num_frames = 10

    framelist = pd.DataFrame.from_dict(
        {"seg_id": ["E001_Neutral_Eyes_Open" for _ in range(num_frames)], "frame_id": [j for j in range(num_frames)]}
    )

    krt_file = "assets/camera_calibration.json"
    krt_dicts = load_camera_calibration(krt_file)
    cameras = list(krt_dicts.keys())

    for f in range(num_frames):
        for c, camera in enumerate(cameras):
            item = getitem(f * len(cameras) + c, framelist, cameras)
            assert item[0] == "E001_Neutral_Eyes_Open"
            assert item[1] == f
            assert item[2] == camera
