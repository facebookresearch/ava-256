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
from typing import List, Optional, Set, Tuple, TypeVar, Union

import einops
import cv2
import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image

from data.utils import HeadsetCapture
from tqdm.auto import tqdm

from zipfile import ZipFile
from zipp import Path as ZipPath


class SingleCaptureDataset(torch.utils.data.IterableDataset):
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
                if segment.startswith("neutral"):
                    segment_name = segment.split("-")[0]

                    # Get the middle frame
                    seg_frame_count = len([elem for elem in self.framelist["seg_id"] if elem.startswith(segment_name)])
                    self._neutral_img = self.get_img_by_segment_and_frame(segment, seg_frame_count//2)
                    return self._neutral_img
            else:
                raise ValueError("Unable to find any neutral images")

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
        return torch.from_numpy(imgs_arr)

    def get_assets_by_id(self, idx: int) -> np.ndarray:
        # Get the segment and frame
        segment, frame = self.framelist["seg_id"][idx], self.framelist["frame_id"][idx]
        segment_name = segment.split("-")[0]
        if self.load_latent_code:
            try:
                latent_code = pickle.loads(ZipPath(
                    self.latent_code_dir / "rosetta_correspondences.zip",
                    f"{self.capture.sid}-{segment}_{segment_name}_{frame}"
                ).read_bytes())["expression"]
            except FileNotFoundError:
                return None
        else:
            latent_code = np.zeros((self.self.latent_code_dim,))

        return {
            "headset_cam_img": self.get_img_by_segment_and_frame(segment, frame),
            "cond_headset_cam_img": self.neutral_img,
            "gt_latent_code": torch.from_numpy(latent_code)
        }

    def __iter__(self):
        while True:
            idx = np.random.randint(0, len(self.framelist))
            yield self.get_assets_by_id(idx)

    def get_img_size(self) -> Tuple[int, int]:
        return (self.height, self.width)


class MultiCaptureDataset(torch.utils.data.ChainDataset):
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
        all_dataset = [
            SingleCaptureDataset(cap, cap_dir, code_dir, downsample, time_downsample, cameras_specified)
            for cap, cap_dir, code_dir in tqdm(zip(captures, directories, latent_code_directories), total=len(captures))
        ]
        super().__init__(all_dataset)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/shaojieb/rsc/ava-256")
    from utils import train_headset_csv_loader
    from data.ava_dataset import none_collate_fn
    from torch import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    base_dir = "/uca/julieta/oss/hmc"
    latent_dir = "/uca/leochli/oss/ava256_udmapping/render_expression_regressor"
    csv_path = "/home/shaojieb/rsc/ava-256/256_ids_enc.csv"
    train_captures, train_dirs = train_headset_csv_loader(base_dir, csv_path, nids=256)
    _, train_latent_code_dirs = train_headset_csv_loader(latent_dir, csv_path, nids=256)
    dataset = MultiCaptureDataset(train_captures[3:5], train_dirs[3:5], train_latent_code_dirs[3:5], downsample=2.083)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        drop_last=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=10,
        collate_fn=none_collate_fn,
    )

    print("Start loading data...")
    for i, data in tqdm(enumerate(dataloader), total=1000):
        if i == 0:
            print(data["headset_cam_img"].shape)
        if i >= 1000:
            break