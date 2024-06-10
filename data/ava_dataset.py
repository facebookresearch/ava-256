import bisect
import io
import logging
import logging.handlers
import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union

import einops
import numpy as np
import pandas as pd
import pillow_avif
import torch.utils.data
from PIL import Image
from plyfile import PlyData
from torch import multiprocessing as mp
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from zipp import Path as ZipPath

from data.utils import MugsyCapture, get_framelist_neuttex_and_neutvert, getitem
from utils import load_camera_calibration

mp.set_start_method("spawn", force=True)

logger = logging.getLogger("ghsv2_airstore_dataset")
logger.setLevel(logging.DEBUG)
logger.propagate = False
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

T = TypeVar("T")


def none_collate_fn(items: List[T]) -> Optional[torch.Tensor]:
    """Modified form of :func:`torch.utils.data.dataloader.default_collate`
    that will strip samples from the batch if they are ``None``."""
    items = [item for item in items if item is not None]
    return default_collate(items) if len(items) > 0 else None


class MultiCaptureDataset(torch.utils.data.Dataset):
    """
    Dataset with CA2 assets for multiple captures

    Args:
        captures: The unique identifiers of the mugsy captures in this dataset
        downsample: Downsampling factor to use when loading images
    """

    def __init__(
        self,
        captures: List[MugsyCapture],
        directories: List[str],
        downsample: int = 4,
        cameras_specified: List[str] = None,
    ):
        super().__init__()

        self.captures = captures
        self.dirs = directories
        self.downsample = downsample
        self.height, self.width = (4096 // downsample, 2668 // downsample)

        # Tuples with all the identifiers of this capture, used in ddp-train
        self.identities = captures

        # Load the single-capture datasets
        self.single_capture_datasets = OrderedDict()
        for capture, capture_dir in tqdm(
            zip(captures, directories),
            desc="Loading single id captures",
            total=len(captures),
        ):
            self.single_capture_datasets[capture] = SingleCaptureDataset(
                capture, capture_dir, downsample, cameras_specified
            )

        # Dataset lengths
        self.cumulative_sizes = np.cumsum([len(x) for x in self.single_capture_datasets.values()])
        self.total_len = self.cumulative_sizes[-1]

        # Compute cross-id normalization stats, and overwrite them in the underlying datasets
        self.texmean, self.texstd = self.get_texture_norm_stats()
        self.vertmean, self.vertstd = self.get_mesh_vert_stats()
        for capture, dataset in self.single_capture_datasets.items():
            dataset.texmean = self.texmean
            dataset.texstd = self.texstd
            dataset.vertmean = self.vertmean
            dataset.vertstd = self.vertstd

        # TODO(julieta) merge all camera names?

    def get_texture_norm_stats(self) -> Tuple[np.ndarray, float]:
        """
        Calculate the texture mean and variance across all subdatasets.
        Technically wrong since we just compute the mean of the means, but it's good enough
        """
        N = len(self.single_capture_datasets)

        # Mean
        texmean = None
        for capture, dataset in self.single_capture_datasets.items():
            if texmean is None:
                texmean = dataset.texmean.copy()
            else:
                texmean += dataset.texmean
        texmean /= N

        # Stdev
        if N == 1:
            # TODO(julieta) probably wrong?!
            texvar = np.mean((texmean - np.mean(texmean, axis=0, keepdims=True)) ** 2)
        else:
            texvar = 0.0
            for capture, dataset in self.single_capture_datasets.items():
                texvar += np.sum((dataset.texmean - texmean) ** 2)
            texvar /= texmean.size * N

        return texmean, math.sqrt(texvar)

    def get_mesh_vert_stats(self) -> Tuple[np.ndarray, float]:
        """
        Calculate the mesh mean and variance across all subdatasets
        """
        N = len(self.single_capture_datasets)

        # Mean
        vertmean = None
        for capture, dataset in self.single_capture_datasets.items():
            if vertmean is None:
                vertmean = dataset.vertmean.copy()
            else:
                vertmean += dataset.vertmean
        vertmean /= N

        # Stdev
        vertvar, vertvar_mean = 0.0, 0.0
        for capture, dataset in self.single_capture_datasets.items():
            vertvar += np.sum((dataset.vertmean - vertmean) ** 2)
            vertvar_mean += dataset.vertstd**2
        vertvar /= vertmean.size * N
        vertvar += vertvar_mean / N

        return vertmean, math.sqrt(vertvar)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Inspired from PyTorch's ConcatDataset"""

        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        capture = self.captures[dataset_idx]
        sample = self.single_capture_datasets[capture][sample_idx]

        if sample is not None:
            sample["idindex"] = dataset_idx

        return sample

    def __len__(self):
        return self.total_len

    def get_allcameras(self) -> Set[str]:
        """Get all the cameras in this dataset"""
        other_cameras = [x.get_allcameras() for x in self.single_capture_datasets.values()]
        return set().union(*other_cameras)

    def get_img_size(self) -> Tuple[int, int]:
        """Get the size of camera images in this dataset"""
        return (self.height, self.width)


class SingleCaptureDataset(torch.utils.data.Dataset):
    """
    Dataset with Mugsy assets for a single capture

    Args:
        capture: The unique identifier of this mugsy capture
        downsample: Downsampling factor to use when loading images
    """

    def __init__(
        self,
        capture: MugsyCapture,
        directory: str,
        downsample: int = 4,
        cameras_specified: List[str] = None,
    ):
        super().__init__()

        self.capture = capture
        self.dir = Path(directory)
        self.downsample = downsample

        # TODO(julieta) pull full image size elsewhere?
        self.height, self.width = (4096 // downsample, 2668 // downsample)

        # Tuple with all the identifiers of this capture, used in ddp-train
        self.identities = [capture]

        assert self.dir.exists(), f"Dataset directory {self.dir} does not seem to exist"

        # Load krt dictionaries
        krt_file = self.dir / "camera_calibration.json"
        krt_dicts = load_camera_calibration(krt_file)

        self.cameras = list(krt_dicts.keys())
        if cameras_specified is not None:
            self.cameras = [x for x in cameras_specified if x in self.cameras]
            if len(self.cameras) == 0:
                raise ValueError(
                    f"There are no cameras left for {self.capture}; requested cameras: {cameras_specified}"
                )

        # Pre-load krts in user-friendly dictionaries
        self.campos, self.camrot, self.focal, self.princpt = {}, {}, {}, {}
        for cam, krt in krt_dicts.items():
            self.campos[cam] = (-np.dot(krt["extrin"][:3, :3].T, krt["extrin"][:3, 3])).astype(np.float32)
            self.camrot[cam] = (krt["extrin"][:3, :3]).astype(np.float32)
            self.focal[cam] = (np.diag(krt["intrin"][:2, :2]) / downsample).astype(np.float32)
            self.princpt[cam] = (krt["intrin"][:2, 2] / downsample).astype(np.float32)

        self.camera_map = dict()
        for i, cam in enumerate(self.cameras):
            self.camera_map[cam] = i

        # Normalization stats
        texmean = np.asarray(Image.open(self.dir / "uv_image" / "color_mean.png"), dtype=np.float32)
        self.texmean = einops.rearrange(texmean, "h w c -> c h w").astype(np.float32).copy("C")
        self.texstd = float(np.genfromtxt(self.dir / "uv_image" / "color_variance.txt") ** 0.5)
        self.vertmean = np.load(self.dir / "kinematic_tracking" / "registration_vertices_mean.npy")
        self.vertstd = float(
            np.genfromtxt(self.dir / "kinematic_tracking" / "registration_vertices_variance.txt") ** 0.5
        )

        self.framelist, self.neut_avgtex, self.neut_vert = get_framelist_neuttex_and_neutvert(self.dir)

    def fetch_data_from_disk(self, frame_id: str, camera_id: str) -> Optional[Dict[str, Union[np.ndarray, int, str]]]:
        try:
            # Camera image
            path = ZipPath(
                self.dir / "image" / f"cam{camera_id}.zip",
                f"cam{camera_id}/{int(frame_id):06d}.avif",
            )
            img_bytes = path.read_bytes()
            img = Image.open(io.BytesIO(img_bytes))
            img = img.resize((self.width, self.height))  # Make of appropriate size
            img = np.asarray(img)
            img = einops.rearrange(img, "h w c -> c h w").astype(np.float32)

            # Mesh
            path = ZipPath(
                self.dir / "kinematic_tracking" / "registration_vertices.zip",
                f"{int(frame_id):06d}.ply",
            )
            ply_bytes = path.read_bytes()
            ply_bytes = io.BytesIO(ply_bytes)
            # verts, _ = p3d.io.load_ply(io.BytesIO(ply_bytes))
            plydata = PlyData.read(ply_bytes)
            verts = plydata["vertex"].data
            verts = np.array([list(element) for element in verts])

            # Average texture
            path = ZipPath(self.dir / "uv_image" / "color.zip", f"color/{int(frame_id):06d}.avif")
            avgtex_bytes = path.read_bytes()
            avgtex = np.asarray(Image.open(io.BytesIO(avgtex_bytes)))
            avgtex = einops.rearrange(avgtex, "h w c -> c h w").astype(np.float32)

            # Head pose (global transform of the person's head)
            path = ZipPath(self.dir / "head_pose" / "head_pose.zip", f"{int(frame_id):06d}.txt")
            headpoose_bytes = path.read_bytes()
            headpose = np.loadtxt(io.BytesIO(headpoose_bytes), dtype=np.float32)

            if any(i is None for i in (img, verts, avgtex, headpose)):
                raise ValueError(f"Some of fetched data is None for {frame_id}-{camera_id}")

        except Exception as e:
            # logger.error(f"Error loading data: {e}")
            logger.exception(e)
            return None

        # pixelcoords
        px, py = np.meshgrid(
            np.arange(self.width).astype(np.float32),
            np.arange(self.height).astype(np.float32),
        )
        pixelcoords = np.stack((px, py), axis=-1)

        return dict(
            # krt info
            camrot=np.dot(headpose[:3, :3].T, self.camrot[camera_id].T).T,
            campos=np.dot(headpose[:3, :3].T, self.campos[camera_id] - headpose[:3, 3]),
            focal=self.focal[camera_id],
            princpt=self.princpt[camera_id],
            modelmatrix=np.eye(4, dtype=np.float32),
            # Encoder inputs
            avgtex=(avgtex - self.texmean) / self.texstd,
            verts=(verts - self.vertmean) / self.vertstd,
            neut_avgtex=(self.neut_avgtex - self.texmean) / self.texstd,
            neut_verts=(self.neut_vert - self.vertmean) / self.vertstd,
            # Select pixels to evalaute ray marching on
            pixelcoords=pixelcoords,
            # Indexing for background/color modeling
            idindex=0,
            camindex=self.camera_map[camera_id],  # TODO handle for multi-id
            # Other
            image=img,
            headpose=headpose,
            frameid=frame_id,
            cameraid=camera_id,
            # id cond info
            validinput=True,
            imagemask=np.ones((1, self.height, self.width), dtype=np.float32),
        )

    def __getitem__(self, idx: int) -> Optional[Dict[str, Union[np.ndarray, int, str]]]:
        return self.fetch_data_from_disk(*getitem(idx, self.framelist, self.cameras)[1:])

    def __len__(self):
        return len(self.cameras) * len(self.framelist)

    ### Methods added for compat with previous version of dataset. Might want to revisit these
    def get_allcameras(self) -> Set[str]:
        return set(self.cameras)

    def get_img_size(self) -> Tuple[int, int]:
        return (self.height, self.width)
