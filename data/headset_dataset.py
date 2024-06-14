# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataloader for loading headset data.
"""

# /uca/julieta/oss/hmc

import io
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image

from data.utils import HeadsetCapture

from zipfile import ZipFile
from zipp import Path as ZipPath


class SingleCaptureDataset(torch.utils.data.Dataset):
    """
    Dataset with headset images for a single capture

    Args:
        capture: The unique identifier of this headset capture
        downsample: Downsampling factor to use when loading images
    """

    def __init__(
        self,
        capture: HeadsetCapture,
        directory: str,
        latent_code_directory: Optional[str] = None,
        downsample: int = 1,  # no downsample
        time_downsample: int = 1,  # no time downsample
        cameras_specified: List[str] = ["left-eye-atl-temporal", "right-eye-atl-temporal", "left-mouth", "right-mouth"],
    ):
        super().__init__()

        self.capture = capture
        self.dir = Path(directory)
        self.latent_code_dir = Path(latent_code_directory)
        self.load_latent_code = self.latent_code_dir.exists()
        self.downsample = downsample
        self.time_downsample = time_downsample
        self.height, self.width = int(400 / downsample), int(400 / downsample)
        self.latent_code_dim = 256
        assert self.dir.exists(), f"Dataset directory {self.dir} does not seem to exist"

        # Filter cameras
        self.cameras = [
            "left-eye-atl-temporal",
            "right-eye-atl-temporal",
            "left-mouth",
            "right-mouth",
            "cyclop",
        ]
        if cameras_specified is not None:
            self.cameras = [x for x in cameras_specified if x in self.cameras]
            if len(self.cameras) == 0:
                raise ValueError(
                    f"There are no cameras left for {self.capture}; requested cameras: {cameras_specified}"
                )

        # Build or load framelist
        framelist_path = self.dir / "frame_list.csv"
        if framelist_path.exists():
            self.framelist = pd.read_csv(framelist_path)
        else:
            self.framelist = self.build_framelist()
            self.framelist.to_csv(framelist_path, index=False)

        # Time downsample, normally every 4th or 8th frame
        self.framelist = self.framelist[:: self.time_downsample]

    def build_framelist(self) -> pd.DataFrame:
        """
        Build a dataframe with all the frames in this capture.
        The columns are: seg_id,frame_id
        """
        # eg vrs_file_AAN112-raised_eyebrows-2023-04-05-17-52-51
        folders = sorted([x for x in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, x))])

        seg_ids, frame_ids = [], []
        for folder in folders:
            # zip file with images
            images_zip = ZipFile(self.dir / folder / "image.zip")

            # Name of segment; eg, raised_eyebrows-2023-04-05-17-52-51
            segment = folder[len("vrs_file_XXXYYY-"):]

            image_names = images_zip.namelist()

            assert len(image_names) % 10 == 0, f"There should be a multiple of 10 images, but found {len(image_names)}"
            image_numbers = np.unique([int(x.split("-")[0]) for x in image_names])

            seg_ids.extend([segment] * len(image_numbers))
            frame_ids.extend(image_numbers)

        df = pd.DataFrame({"seg_id": seg_ids, "frame_id": frame_ids})
        return df

    @property
    def neutral_img(self) -> np.ndarray:
        if getattr(self, "_neutral_img", None) is not None:
            return self._neutral_img
        else:
            for idx, segment in enumerate(self.framelist["seg_id"]):
                if segment.startswith("neutral") or segment.startswith("EXP_neutral"):
                    segment_name = segment.split("-")[0]

                    # Get the middle frame
                    seg_frame_count = len([s for s in self.framelist["seg_id"] if s.split("-")[0] == segment_name])
                    self._neutral_img = self.get_img_by_segment_and_frame(segment, seg_frame_count//2)
                    return self._neutral_img
            else:
                raise ValueError(f"Unable to find any neutral images. Subject: {self.capture.sid}")

    def get_img_by_segment_and_frame(self, segment: str, frame: int) -> np.ndarray:
        imgs = []
        for camera in self.cameras:
            # Load image
            folder = f"vrs_file_{self.capture.sid}-{segment}"
            img_path = ZipPath(self.dir / folder / "image.zip", f"{frame}-{camera}.png")
            img_bytes = img_path.read_bytes()

            arr = np.asarray(Image.open(io.BytesIO(img_bytes)))
            arr = cv2.resize(arr, (self.height, self.width), interpolation=cv2.INTER_AREA)
            if arr.ndim == 2:
                arr = arr[..., None]

            # HWC, keep single channel
            arr = arr.transpose(2, 0, 1).astype(np.float32)[:1]
            imgs.append(arr)
        imgs_arr = np.stack(imgs, axis=0)
        return (torch.from_numpy(imgs_arr) / 255. - 0.5) * 2

    def get_assets_by_id(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get the segment and frame
        segment, frame = self.framelist["seg_id"][idx], self.framelist["frame_id"][idx]
        segment_name = segment.split("-")[0]
        if self.load_latent_code:
            entry_name = f"{self.capture.sid}-{segment}_{segment_name}_{frame}"
            try:
                latent_code = pickle.loads(ZipPath(
                    self.latent_code_dir / "rosetta_correspondences.zip",
                    entry_name
                ).read_bytes())["expression"]
            except FileNotFoundError:
                return None
        else:
            print("Using placeholder for GT code.")
            latent_code = np.zeros((self.self.latent_code_dim,))

        return {
            "headset_cam_img": self.get_img_by_segment_and_frame(segment, frame),
            "cond_headset_cam_img": self.neutral_img[:, None],
            "gt_latent_code": torch.from_numpy(latent_code),
            "index": {
                "ident": self.capture.sid,
                "segment": segment_name,
                "frame": frame
            }
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.get_assets_by_id(idx)

    def __len__(self) -> int:
        return len(self.framelist)

    def get_random_frame(self):
        idx = np.random.randint(0, len(self.framelist))
        return self.get_assets_by_id(idx)

    def get_img_size(self) -> Tuple[int, int]:
        return (self.height, self.width)


class MultiCaptureDataset(torch.utils.data.IterableDataset):
    """
    Dataset with headset images for multiple captures

    Args:
        captures: A list of unique identifiers of the headset captures in this dataset
        directories: A list of the directory paths for each capture
        downsample: Downsampling factor to use when loading images
        time_downsample: Time downsampling factor to use when loading images
        cameras_specified: A list of the headset camera names to load
    """

    def __init__(
        self,
        captures: List[HeadsetCapture],
        directories: List[str],
        latent_code_directories: Optional[List[Union[str, Optional[str]]]],
        downsample: int = 4,
        time_downsample: int = 1,  # no time downsample
        cameras_specified: List[str] = ["left-eye-atl-temporal", "right-eye-atl-temporal", "left-mouth", "right-mouth"],
    ):
        self.identities = captures
        if latent_code_directories is None:
            latent_code_directories = [None] * len(captures)
        self.datasets = [
            SingleCaptureDataset(cap, cap_dir, code_dir, downsample, time_downsample, cameras_specified)
            for cap, cap_dir, code_dir in zip(captures, directories, latent_code_directories)
        ]

    def __iter__(self):
        # TODO: Make this a map-style dataset and use a random sampler with replacement
        while True:
            dataset_idx = np.random.randint(0, len(self.datasets))
            dataset = self.datasets[dataset_idx]
            yield dataset.get_random_frame()
