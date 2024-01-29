"""
Dataset of decompressed avatar images from a filesystem.

TODO(julieta) make sure this works on windows
TODO(julieta) make sure this works on any POSIX
"""

import bisect
import logging
import logging.handlers
import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import einops
import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image
from torch import multiprocessing as mp
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from data.mugsy_dataset import MugsyCapture
from utils import load_krt

mp.set_start_method("spawn", force=True)

logger = logging.getLogger("ghsv2_airstore_dataset")
logger.setLevel(logging.DEBUG)
logger.propagate = False
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


# Folder has `/uca2`` for uca2 assets and `/minisis` for minisis assets
os.environ["RSC_AVATAR_METADATA_PATH"] = "/uca/uca2-meta/"


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
    ):
        super().__init__()

        self.captures = captures
        self.dirs = directories
        self.downsample = downsample
        self.height, self.width = (4096 // downsample, 2668 // downsample)

        # Tuples with all the identifiers of this capture, used in ddp-train
        self.identities = captures

        # Create many single-capture datasets
        self.single_capture_datasets = OrderedDict()
        for capture, dir in tqdm(zip(captures, directories), desc="Loading single id captures"):
            self.single_capture_datasets[capture] = SingleCaptureDataset(capture, dir, downsample)

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

    ### Methods added for compat with previous version of dataset. Might want to revisit these
    def get_allcameras(self) -> Set[str]:
        other_cameras = [x.get_allcameras() for x in self.single_capture_datasets.values()]
        return set().union(*other_cameras)

    def get_img_size(self) -> Tuple[int, int]:
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
    ):
        super().__init__()

        self.capture = capture
        self.dir = Path(directory)
        self.downsample = downsample

        # TODO(julieta) pull full image size elsewhere?
        self.height, self.width = (4096 // downsample, 2668 // downsample)

        # Tuple with all the identifiers of this capture, used in ddp-train
        self.identities = [capture]

        assert os.path.exists(self.dir), f"Dataset directory {self.dir} does not seem to exist"

        # Load krt dictionaries
        krt_file = self.dir / "KRT"
        krt_dicts = load_krt(krt_file)
        self.cameras = list(krt_dicts.keys())

        # Pre-load krts in user-friendly dictionaries
        self.campos, self.camrot, self.focal, self.princpt = {}, {}, {}, {}
        for cam, krt in krt_dicts.items():
            self.campos[cam] = (-np.dot(krt["extrin"][:3, :3].T, krt["extrin"][:3, 3])).astype(np.float32)
            self.camrot[cam] = (krt["extrin"][:3, :3]).astype(np.float32)
            self.focal[cam] = (np.diag(krt["intrin"][:2, :2]) * 2 / downsample).astype(np.float32)
            self.princpt[cam] = (krt["intrin"][:2, 2] * 2 / downsample).astype(np.float32)

        self.camera_map = dict()
        for i, cam in enumerate(self.cameras):
            self.camera_map[cam] = i

        # Load frame list; ie, (segment, frame) pairs
        frame_list_path = self.dir / "frame_list.txt"
        self.framelist = pd.read_csv(frame_list_path, names=["seg_id", "frame_id"], dtype=str, delim_whitespace=True)

        # Filter by segments
        segments_to_keep = ["E001_Neutral_Eyes_Open", "E057_Cheeks_Puffed", "E061_Lips_Puffed"]
        self.framelist = self.framelist[self.framelist["seg_id"].isin(segments_to_keep)]

        # Normalization stats
        texmean = np.asarray(Image.open(self.dir / "tex_mean.png"), dtype=np.float32)
        self.texmean = einops.rearrange(texmean, "h w c -> c h w").astype(np.float32).copy("C")
        self.texstd = float(np.genfromtxt(self.dir / "tex_var.txt") ** 0.5)
        self.vertmean = np.fromfile(self.dir / "vert_mean.bin", dtype=np.float32).reshape(-1, 3)
        self.vertstd = float(np.genfromtxt(self.dir / "vert_var.txt") ** 0.5)

        # Neutral conditioning
        # neutral_segment = "EXP_neutral_peak"  # TODO(julieta) choose from a list of potential neutral segments
        neutral_segment = "E001_Neutral_Eyes_Open"
        neut_framelist = self.framelist.loc[self.framelist["seg_id"] == neutral_segment].values.tolist()
        # vlist, tlist = [], []
        for neut_seg, neut_frame in neut_framelist:
            verts = np.fromfile(
                self.dir / f"tracked_mesh/{neut_seg}/{neut_frame}.bin",
                dtype=np.float32,
            ).reshape(-1, 3)

            tex_path = self.dir / f"unwrapped_uv_1024/{neut_seg}/average/{neut_frame}.png"
            tex = np.asarray(Image.open(tex_path))
            tex = einops.rearrange(tex, "h w c -> c h w").astype(np.float32)

            # vlist.append(verts)
            # tlist.append(tex)
            # NOTE(julieta) only load one since this might be causing OOM issues
            self.neut_avgtex = tex
            self.neut_vert = verts
            break

        # assert len(tlist) > 0, "neut_verts should not be empty"
        # self.neut_avgtexs = tlist
        # self.neut_verts = vlist

        if self.neut_avgtex is None:
            raise ValueError("Not able to find any neutral average textures")
        if self.neut_vert is None:
            raise ValueError("Not able to find any neutral vertices")

        # # TODO(julieta) sample randomly from the ones above?
        # self.neut_avgtex = tlist[0]
        # self.neut_vert = vlist[0]

        # Load background images from NFS
        # self.bg_model_path = (
        #     f"{self.metadata_dir}/ca2/learn_bkg_per_camera/{capture.sid.upper()}_{capture.mcd}--0000_codec/"
        # )
        # for cam_id in self.cameras:
        #     bg_img = Image.open(f"{bg_model_path}/bg_{cam_id}.png")
        #     bg_img = einops.rearrange(np.asarray(bg_img), 'h w c -> c h w').astype(np.float32)  # bg image is current in 1024x667 resolution
        #     self.bg_imgs[cam_id] = bg_img
        # print(f"Loaded background images in {time.time() - st} seconds")

    def fetch_data_from_disk(
        self, segment: str, frame_id: str, camera_id: str
    ) -> Optional[Dict[str, Union[np.ndarray, int, str]]]:
        try:
            # Camera image
            path = self.dir / "images" / segment / camera_id / f"{frame_id}.png"
            img = Image.open(path)
            img = img.resize((self.width, self.height))  # Make of appropriate size
            img = np.asarray(img)
            img = einops.rearrange(img, "h w c -> c h w").astype(np.float32)

            # Mesh
            path = self.dir / "tracked_mesh" / segment / f"{frame_id}.bin"
            verts = np.fromfile(path, dtype=np.float32).reshape(-1, 3)

            # Average texture
            path = self.dir / "unwrapped_uv_1024" / segment / "average" / f"{frame_id}.png"
            avgtex = np.asarray(Image.open(path))
            avgtex = einops.rearrange(avgtex, "h w c -> c h w").astype(np.float32)

            # Head pose (global transform of the person's head)
            path = self.dir / "tracked_mesh" / segment / f"{frame_id}_transform.txt"
            headpose = np.loadtxt(path, dtype=np.float32)

            if any(i is None for i in (img, verts, avgtex, headpose)):
                raise ValueError(f"Some of fetched data is None for {segment}-{frame_id}-{camera_id}")

        except Exception as e:
            logger.error(f"Error loading airstore data: {e}")
            return None

        # pixelcoords
        px, py = np.meshgrid(np.arange(self.width).astype(np.float32), np.arange(self.height).astype(np.float32))
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
            segment=segment,
            # id cond info
            validinput=True,
            imagemask=np.ones((1, self.height, self.width), dtype=np.float32),
        )

    def __getitem__(self, idx: int) -> Optional[Dict[str, Union[np.ndarray, int, str]]]:
        segment_and_frame = self.framelist.iloc[idx]
        segment: str = segment_and_frame.seg_id
        frame: str = segment_and_frame.frame_id
        camera = self.cameras[idx % len(self.cameras)]
        return self.fetch_data_from_disk(segment, frame, camera)

    def __len__(self):
        return len(self.framelist)

    ### Methods added for compat with previous version of dataset. Might want to revisit these
    def get_allcameras(self) -> Set[str]:
        return set(self.cameras)

    def get_img_size(self) -> Tuple[int, int]:
        return (self.height, self.width)
