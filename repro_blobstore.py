"""
Random Access dataset that supports train/val/evaluation
"""
import os
import sys
# sys.path.insert(0, "/uca_transient_a/wenj/care-master-2023-06-16-final")
import cv2
cv2.setNumThreads(0)

import einops
import numpy as np
import scipy.ndimage
import torch.utils.data
import torchvision
import torch.nn.functional as F
from PIL import Image
import copy
import hashlib as hlib
from pathlib import Path
import io

from tqdm import tqdm
import logging
import logging.handlers
import math
import os
import pickle
import random
import sys
from typing import Any, Dict, List, Tuple, Set, Literal, Callable, Optional, Union, TypeVar

import time
import tempfile

from dataclasses import dataclass
from care.data.io import typed
from care.strict.data.io.file_system.airstore_client import register_airstore_in_fsspec

from data.tiled_png_utils import (
    process_dataset_metadata,
    generate_transform,
    process_tile_png_sample,
)

from airstore.blobstore import BlobStore

from care.strict.data.io.typed.file.calib import KRTFile
from care.strict.data.io.typed.file.obj_np import ObjFileNumpy
from care.strict.data.io.typed.file.image import ImageFile
from care.strict.data.io.typed.file.calib import KRTFile
from care.strict.data.io.typed.file.stdlib import PickleFile


# Use cfg node as a quick but dirty trick for fast prototyping
from yacs.config import CfgNode
import pandas as pd
from torch.utils.data.dataloader import default_collate

from torch import multiprocessing as mp
mp.set_start_method("spawn", force=True)

logger = logging.getLogger('ghsv2_airstore_dataset')
logger.setLevel(logging.DEBUG)
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


class SingleCaptureDataset(torch.utils.data.Dataset):
    """
    Dataset with MGR images and assets for a single subject

    Args:
        capture: The unique identifier of this MGR capture
        downsample: Downsampling factor to use when loading images
    """

    def __init__(
        self,
        capture: str,
        downsample: int = 1,
    ):
        super().__init__()

        # TODO(julieta) do not overwrite if already set
        os.environ["DB_CACHE_DIR"] = "/uca_transient_a/db_indices/"
        register_airstore_in_fsspec()

        # assert len(capture) == len("1000083317329676"), f"Capture should have {len('1000083317329676')} characters, not {len(capture)}"

        self.capture = capture
        self.downsample = downsample
        self.height, self.width = (960 // downsample, 720 // downsample)

        # Tuple with all the identifiers of this capture, used in ddp-train
        self.identities = [capture]

        # Frames and assets tables
        self.images_table_name = "codec_avatar_mgr_12k_frames_no_user_data"
        self.assets_table_name = "codec_avatar_mgr_12k_assets_v1_no_user_data"
        # Per-capture assets
        self.per_capture_assets = "codec_avatar_mgr_12k_per_subject_no_user_data"

        # LOAD PER-CAPTURE ASSETS

        # Frame list
        url = f"airstoreds://{self.per_capture_assets}/frame_list?subject_id={self.capture}"
        framelist_bytes = typed.load(url, extension="bin")
        self.framelist = pd.read_csv(io.BytesIO(framelist_bytes), names=["seg_id", "frame_id"], delim_whitespace=True)

        # ID params
        url = f"airstoreds://{self.per_capture_assets}/id_params?subject_id={self.capture}"
        id_params = typed.load(url, extension="pkl")
        # id_params is a dictionary with fields:
        #  'l_eye_center',
        #  'l_eye_rotation_center',
        #  'l_eye_shape_params',
        #  'neck_orig',
        #  'o_v_rvec',
        #  'r_eye_center',
        #  'r_eye_rotation_center',
        #  'r_eye_shape_params',
        #  'z_id'

        # url = f"airstoreds://{self.per_capture_assets}/neu_geo?subject_id={self.capture}"
        # self.neut_geo = typed.load(url, extension="obj")  # Has v, vt, vi, vti

        # # Neutral texture -- didn't seem much different from v2?
        # url = f"airstoreds://{self.per_capture_assets}/neu_tex?subject_id={self.capture}"
        # neut_tex = typed.load(url, extension="png")
        # self.neut_tex = einops.rearrange(neut_tex, "h w c -> c h w").astype(np.float32)

        # TODO(julieta) load torso tex and geo?
        # neu_torso_geo	blob	subject_zid
        # neu_torso_text	blob	subject_id
        self.cameras = [0]


        self.texmean = einops.rearrange(np.array([0.485, 0.456, 0.406]) * 255, "c -> c 1 1").astype(np.float32)
        self.texstd  = einops.rearrange(np.array([0.229, 0.224, 0.225]) * 255, "c -> c 1 1").astype(np.float32)

        # TODO(julieta) get vertmean and vertstd somehow?
        self.vertmean = np.load("/checkpoint/avatar/julietamartinez/vertmean.npy").astype(np.float32)
        self.vertstd = np.load("/checkpoint/avatar/julietamartinez/vertstd.npy").astype(np.float32).item()

        # # Neutral conditioning
        # neut_framelist = self.framelist.loc[self.framelist["seg_id"] == "static_neutral"].values.tolist()
        # for neut_seg, neut_frame in neut_framelist:

        #     # Load verts
        #     url = f"airstoreds://{self.assets_table_name}/fit?subject_id={self.capture}&frame_id={neut_frame}&segment={neut_seg}"
        #     fit = typed.load(url, extension="pkl")
        #     verts = fit["verts"]

        #     # Load tex
        #     url = f"airstoreds://{self.images_table_name}/image?subject_id={self.capture}&frame_id={neut_frame}&segment={neut_seg}"
        #     tex = typed.load(url, extension="png")
        #     tex = einops.rearrange(tex, "h w c -> c h w").astype(np.float32)

        #     # vlist.append(verts)
        #     # tlist.append(tex)
        #     # NOTE(julieta) only load one since this might be causing OOM issues
        #     self.neut_avgtex = tex
        #     self.neut_vert = verts
        #     break
        # else:
        #     raise ValueError("Did not find any neutral frames")


        # self.krt = None
        # self.focal = None
        # self.princpt = None

        # with BlobStore() as blobstore:
        #     blobstore.start()
        #     krt_bytes = blobstore.get(bucket="avatar", key=f"mgr/{self.capture}/Segs/KRT")
        #     self.krt = KRTFile.deserialize(krt_bytes, format="krt")
        #     # krt has is a nested dict with two keys: 'iPhone_rgb' and 'iPhone_depth', each sub-dict has
        #     # 'intrin', 'extrin', 'dist', 'model', 'height', 'width'
        #     self.focal   = self.krt["iPhone_rgb"]["intrin"][:2, :2]
        #     self.princpt = self.krt["iPhone_rgb"]["intrin"][:2,  2]

        #     # TODO(julieta) load v2 neut tex
        #     neut_geo_bytes = blobstore.get(bucket="avatar", key=f"mgr/{self.capture}/Segs/neu_geo_v2.obj")
        #     neut_geo = ObjFileNumpy.deserialize(neut_geo_bytes)
        #     self.neut_vert = neut_geo["v"].astype(np.float32)

        #     # neut_tex
        #     neut_tex_bytes = blobstore.get(bucket="avatar", key=f"mgr/{self.capture}/Segs/neu_tex_v2.png")
        #     neut_tex = np.asarray(Image.open(io.BytesIO(neut_tex_bytes)))
        #     self.neut_avgtex = einops.rearrange(neut_tex, "h w c -> c h w").astype(np.float32)


        # Load KRT from NFS
        self.asset_path = f"/checkpoint/avatar/julietamartinez/mgr/assets/{self.capture}/Segs"
        self.krt = typed.load(f"{self.asset_path}/KRT", extension="krt")
        self.focal   = self.krt["iPhone_rgb"]["intrin"][:2, :2]
        self.princpt = self.krt["iPhone_rgb"]["intrin"][:2,  2]


    def fetch_airstore_data(self, frame_id: str, segment: str) -> Optional[Dict[str, np.ndarray]]:

        # NOTE(julieta) the rows that get pulled from the assets table are documented at
        # https://www.internalfb.com/intern/wiki/RL/RL_Research/Pittsburgh/Engineering/Onboarding/Avatar_RSC_Manual/Supported_Datasets/MGR_Dataset/

        try:

            # Camera image
            url = f"airstoreds://{self.images_table_name}/image?subject_id={self.capture}&frame_id={frame_id}&segment={segment}"
            img = typed.load(url, extension="png")
            # TODO(julieta) resize before converting to np.array
            # img = img.resize((self.width, self.height))
            img = einops.rearrange(img, "h w c -> c h w").astype(np.float32)

            # # Depth
            # url = f"airstoreds://{self.assets_table_name}/depth?subject_id={self.capture}&frame_id={frame_id}&segment={segment}"
            # depth = typed.load(url, extension="npbin", dtype=np.float32)

            # # Keypoints (243 points)
            url = f"airstoreds://{self.assets_table_name}/kpts?subject_id={self.capture}&frame_id={frame_id}&segment={segment}"
            kpts = typed.load(url, extension="npbin", dtype=np.float32).reshape(-1, 3)  # Shape 243 x 3. Third column might be confidence?

            # # Average texture
            url = f"airstoreds://{self.assets_table_name}/tex?subject_id={self.capture}&frame_id={frame_id}&segment={segment}"
            avgtex = typed.load(url, extension="png")
            avgtex = einops.rearrange(avgtex, "h w c -> c h w").astype(np.float32)

            # # Fit
            url = f"airstoreds://{self.assets_table_name}/fit?subject_id={self.capture}&frame_id={frame_id}&segment={segment}"
            fit = typed.load(url, extension="pkl")
            rot = fit["rot"]
            trans = fit["trans"]
            verts = fit["verts"]
            torso_verts = fit["torso_verts"]

            R, _ = cv2.Rodrigues(rot)

            # Load neutral assets here because otherwise the assets of 12k ids can't be held in memory
            # with BlobStore() as blobstore:
            #     blobstore.start()

            #     krt_bytes = blobstore.get(bucket="avatar", key=f"mgr/{self.capture}/Segs/KRT")
            #     if self.krt is None:
            #         self.krt = KRTFile.deserialize(krt_bytes, format="krt")
            #         self.focal   = self.krt["iPhone_rgb"]["intrin"][:2, :2]
            #         self.princpt = self.krt["iPhone_rgb"]["intrin"][:2,  2]

            #     # neut tex
            #     neut_geo_bytes = blobstore.get(bucket="avatar", key=f"mgr/{self.capture}/Segs/neu_geo_v2.obj")
            #     neut_geo = ObjFileNumpy.deserialize(neut_geo_bytes)
            #     neut_vert = neut_geo["v"].astype(np.float32)

            #     # neut_tex
            #     neut_tex_bytes = blobstore.get(bucket="avatar", key=f"mgr/{self.capture}/Segs/neu_tex_v2.png")
            #     neut_tex = np.asarray(Image.open(io.BytesIO(neut_tex_bytes)))
            #     neut_avgtex = einops.rearrange(neut_tex, "h w c -> c h w").astype(np.float32)

            # Load from NFS
            # neut_geo = typed.load(f"{self.asset_path}/neu_geo_v2.obj", extension="obj")
            # neut_vert = neut_geo["v"].astype(np.float32)

            # neut_tex = typed.load(f"{self.asset_path}/neu_tex_v2.png", extension="png")
            # neut_avgtex = einops.rearrange(neut_tex, "h w c -> c h w").astype(np.float32)

            # Torso conditioning
            torso_neut_geo = typed.load(f"{self.asset_path}/neu_torso_geo_v2.obj", extension="obj")
            torso_neut_vert = torso_neut_geo["v"].astype(np.float32)

            torso_neut_tex = typed.load(f"{self.asset_path}/neu_torso_tex_v2.png", extension="png")
            torso_neut_avgtex = einops.rearrange(torso_neut_tex, "h w c -> c h w").astype(np.float32)


            if any(i is None for i in (img, verts, avgtex, neut_vert, neut_avgtex)):
                raise ValueError(f"Some of fetched data is None for {self.assets_table}-{frame_id}-{segment}")

        except Exception as e:
            logger.error(f"Error loading airstore data {frame_id=} {segment=}: {e}")
            return None

        # pixelcoords
        px, py = np.meshgrid(np.arange(self.width).astype(np.float32), np.arange(self.height).astype(np.float32))

        return dict(
            image=img,
            verts=(torso_verts - self.vertmean) / self.vertstd,
            avgtex=(avgtex - self.texmean) / self.texstd,
            # headpose=headpose,
            frameid=frame_id,
            cameraid=0,  # There is only one camera
            # id cond info
            neut_verts=(torso_neut_vert - self.vertmean) / self.vertstd,
            neut_avgtex=(torso_neut_avgtex - self.texmean) / self.texstd,
            # krt info
            camrot=R,
            campos=(-R.T @ trans[:, None]).astype(np.float32)[:, 0],
            focal=np.asarray([self.focal[0, 0], self.focal[1, 1]], dtype=np.float32),
            princpt=self.princpt.astype(np.float32),
            modelmatrix=np.eye(4, dtype=np.float32),
            validinput=True,
            pixelcoords=np.stack((px, py), axis=-1),
            # TODO(julieta) use mask as in previous dataset
            imagemask=np.ones((1, self.height, self.width), dtype=np.float32),
            # ididx and camidx
            idindex=0,
            camindex=0,  # Single dataset for
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # TODO(julieta) select from the tsv, or pass the frame and camera directly. This random sampling is bad design.
        frame = self.framelist.iloc[idx].frame_id
        segment = self.framelist.iloc[idx].seg_id
        print(frame, segment)
        return self.fetch_airstore_data(int(frame), segment)

    def __len__(self):
        return len(self.framelist)

    ### Methods added for compat with previous version of dataset. Might want to revisit these
    def get_allcameras(self) -> Set[str]:
        return set(self.cameras)

    def get_img_size(self) -> Tuple[int, int]:
        return (self.height, self.width)


if __name__ == "__main__":
    a = SingleCaptureDataset("1000275597227205")
    b = a[0]
    print(b)
