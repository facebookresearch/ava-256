"""
RSC dataloader. Need another one for reading from SSD/disk.
"""

import bisect
import logging
import logging.handlers
import math
import os
import pickle
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union

import einops
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from torch import multiprocessing as mp
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

try:
    from care.data.io import typed
    from care.strict.data.io.file_system.airstore_client import register_airstore_in_fsspec

    from data.tiled_png_utils import generate_transform, process_dataset_metadata, process_tile_png_sample
except:
    print("Could not load CARE, mugsy dataset might not work!")

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


@dataclass(frozen=True)
class MugsyCapture:
    """Unique identifier for a Mugsy capture"""

    mcd: str  # Mugsy capture date in 'yyyymmdd' format, eg `20210223`
    mct: str  # Mugsy capture time in 'hhmm' format, eg `1023`
    sid: str  # Subject ID, three letters and three numbers, eg `avw368`
    is_relightable: bool = False  # Whether this is a relightable capture


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
        downsample: int = 4,
    ):
        super().__init__()

        self.captures = captures
        self.downsample = downsample
        self.height, self.width = (4096 // downsample, 2668 // downsample)

        # Tuples with all the identifiers of this capture, used in ddp-train
        self.identities = captures

        # Create many single-capture datasets
        self.single_capture_datasets = OrderedDict()
        for capture in tqdm(captures, desc="Loading single id captures"):
            self.single_capture_datasets[capture] = SingleCaptureDataset(capture, downsample)

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

    def get_texture_norm_stats(self):
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

    def get_mesh_vert_stats(self):
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
    Dataset with CA2 assets for a single capture

    Args:
        capture: The unique identifier of this mugsy capture
        downsample: Downsampling factor to use when loading images
    """

    def __init__(
        self,
        capture: MugsyCapture,
        downsample: int = 4,
    ):
        super().__init__()

        self.capture = capture
        self.downsample = downsample
        self.height, self.width = (4096 // downsample, 2668 // downsample)

        # Tuple with all the identifiers of this capture, used in ddp-train
        self.identities = [capture]
        meta_folder_name = f"m--{capture.mcd}--{capture.mct}--{capture.sid.upper()}--GHS"

        # Frames and CA2 assets tables
        if capture.is_relightable:
            self.metadata_dir = f"/uca/uca2-meta/300_relit/{meta_folder_name}"
            self.images_table_name = f"codec_avatar_300_relit_cid_{capture.sid.lower()}_{capture.mcd}_{capture.mct}_relightable_mugsy_frames_payload_no_user_data"
            self.assets_table_name = f"codec_avatar_300_relit_cid_{capture.sid.lower()}_{capture.mcd}_{capture.mct}_uca2_v1_0_assets_no_user_data"
        else:
            self.metadata_dir = f"/uca/uca2-meta/{meta_folder_name}"
            self.images_table_name = (
                f"codec_avatar_{capture.sid.lower()}_{capture.mcd}_{capture.mct}_mugsy_frames_no_user_data"
            )
            self.assets_table_name = (
                f"codec_avatar_{capture.sid.lower()}_{capture.mcd}_{capture.mct}_uca2_v1_0_assets_no_user_data"
            )

        assert os.path.exists(self.metadata_dir), f"Meta data directory not found at: {self.metadata_dir}"

        # Pre-load the cambyte transforms so we can load tiled_png images
        minisis_folder = os.path.join(self.metadata_dir, "minisis")

        cache_dir = f"/uca/julieta/cache/m--{capture.mcd}--{capture.mct}--{capture.sid.upper()}--GHS/"
        cached_cambyte_transforms_file = os.path.join(cache_dir, "cambyte_transforms.pkl")
        cached_krt_dicts_file = os.path.join(cache_dir, "krt_dict.pkl")

        if os.path.exists(cached_cambyte_transforms_file) and os.path.exists(cached_krt_dicts_file):
            print(f"{cached_cambyte_transforms_file} and {cached_krt_dicts_file} exist")

            # Load cambyte transforms
            with open(cached_cambyte_transforms_file, "rb") as f:
                self.cambyte_transforms = pickle.load(f)

            # Load krt dictionaries
            with open(cached_krt_dicts_file, "rb") as f:
                krt_dicts = pickle.load(f)

            self.cameras = list(krt_dicts.keys())

        else:
            print(f"{cached_cambyte_transforms_file} does not exist")

            cambyte_metadata = process_dataset_metadata(minisis_folder, capture.mcd, capture.mct, capture.sid)
            self.cameras = list(cambyte_metadata["krt_dict"].keys())

            self.cambyte_transforms = dict()
            for camera in tqdm(self.cameras, desc=f"Loading cambyte transforms for {self.capture}"):
                transform = generate_transform(cambyte_metadata, camera, downsample_factor=downsample)
                if transform is not None:
                    self.cambyte_transforms[camera] = transform

            # Save if it had not been dumped before
            os.makedirs(cache_dir, exist_ok=True)

            with open(cached_cambyte_transforms_file, "wb") as f:
                pickle.dump(self.cambyte_transforms, f)

            krt_dicts = cambyte_metadata["krt_dict"]
            with open(cached_krt_dicts_file, "wb") as f:
                pickle.dump(krt_dicts, f)

        # Pre-load krts in user-friendly dicts
        self.campos, self.camrot, self.focal, self.princpt = {}, {}, {}, {}
        for cam, krt in krt_dicts.items():
            self.campos[cam] = (-np.dot(krt["extrin"][:3, :3].T, krt["extrin"][:3, 3])).astype(np.float32)
            self.camrot[cam] = (krt["extrin"][:3, :3]).astype(np.float32)
            self.focal[cam] = (np.diag(krt["intrin"][:2, :2]) / downsample).astype(np.float32)
            self.princpt[cam] = (krt["intrin"][:2, 2] / downsample).astype(np.float32)

        self.camera_map = dict()
        for i, cam in enumerate(self.cameras):
            self.camera_map[cam] = i

        if capture.is_relightable:
            frame_list_path = os.path.join(minisis_folder, "frame_list_full_lit_frames_only.txt")
        else:
            frame_list_path = os.path.join(minisis_folder, "frame_list.txt")

        self.framelist = pd.read_csv(frame_list_path, names=["seg_id", "frame_id"], delim_whitespace=True)
        self.frames = sorted(np.unique(self.framelist["frame_id"].values))

        # TODO(julieta) check if this is needed
        self.is_setup = False

        # Normalization stats
        texmean = np.asarray(Image.open(os.path.join(minisis_folder, "codec", "tex_mean.png")), dtype=np.float32)
        self.texmean = einops.rearrange(texmean, "h w c -> c h w").astype(np.float32).copy("C")
        self.texstd = float(np.genfromtxt(os.path.join(minisis_folder, "codec", "tex_var.txt")) ** 0.5)
        self.vertmean = np.fromfile(os.path.join(minisis_folder, "codec", "vert_mean.bin"), dtype=np.float32).reshape(
            -1, 3
        )
        self.vertstd = float(np.genfromtxt(os.path.join(minisis_folder, "codec", "vert_var.txt")) ** 0.5)

        # Neutral conditioning
        neut_framelist = self.framelist.loc[self.framelist["seg_id"] == "EXP_neutral_peak"].values.tolist()
        # vlist, tlist = [], []
        for neut_seg, neut_frame in neut_framelist:
            verts = np.fromfile(
                f"{minisis_folder}/tracked_mesh/{neut_seg}/{neut_frame:06d}.bin", dtype=np.float32
            ).reshape(-1, 3)
            # verts = (verts - self.vertmean) / self.vertstd

            tex = np.asarray(
                Image.open(f"{minisis_folder}/unwrapped_uv_1024/{neut_seg}/average/{neut_frame:06d}.png"),
                dtype=np.uint8,
            )
            tex = einops.rearrange(tex, "h w c -> c h w").astype(np.float32)
            # tex = (tex - self.texmean) / self.texstd

            # vlist.append(verts)
            # tlist.append(tex)
            # NOTE(julieta) only load one since this might be causing OOM issues
            self.neut_avgtex = tex
            self.neut_vert = verts
            break

        # assert len(tlist) > 0, "neut_verts should not be empty"
        # self.neut_avgtexs = tlist
        # self.neut_verts = vlist

        # # TODO(julieta) sample randomly from the ones above?
        # self.neut_avgtex = tlist[0]
        # self.neut_vert = vlist[0]

        # Load background images from NFS
        self.bg_model_path = (
            f"{self.metadata_dir}/ca2/learn_bkg_per_camera/{capture.sid.upper()}_{capture.mcd}--0000_codec/"
        )
        # for cam_id in self.cameras:
        #     bg_img = Image.open(f"{bg_model_path}/bg_{cam_id}.png")
        #     bg_img = einops.rearrange(np.asarray(bg_img), 'h w c -> c h w').astype(np.float32)  # bg image is current in 1024x667 resolution
        #     self.bg_imgs[cam_id] = bg_img
        # print(f"Loaded background images in {time.time() - st} seconds")

    def setup(self) -> None:
        """This func is suppose to be called by each of the worker"""
        register_airstore_in_fsspec()
        self.is_setup = True

    def fetch_airstore_data(self, frame_id: str, camera_id: str) -> Optional[Dict[str, Union[np.ndarray, int, str]]]:
        # NOTE(julieta) the rows that get pulled from the assets table are documented at
        # https://www.internalfb.com/intern/wiki/RL/RL_Research/Pittsburgh/Engineering/Onboarding/Avatar_RSC_Manual/Supported_Datasets/Fully-Lit_CA2/
        #
        # The rest of the assets needed for UCA2 are generated from stages under self.metadata_path/ca2
        #
        # triangulate_mugsy_keypoints
        # dump_mugsy_depth_imgs
        # predict_mugsy_keypoints_segmentations (for both ltt and fgbg)
        # mode_pursuit_headpose (for ca2_geom/headpose)
        # train_neck_vae (for neckrot and neckgeom)
        # mugsy_gaze_vectors

        # Check if the worker is initialized
        if not self.is_setup:
            self.setup()

        if camera_id not in self.cambyte_transforms:
            logger.warning(f"Camera {camera_id} does not have a CamByte transform, skipping.")
            return None

        try:
            # Camera image
            url = f"airstoreds://{self.images_table_name}/frame?frame_id={frame_id}&camera={camera_id}"  # noqa: B950
            img_tiled_bytes = typed.load(url, extension="bin")
            img = self.cambyte_transforms[camera_id](img_tiled_bytes)
            img = einops.rearrange(img, "h w c -> c h w")
            img = img.astype(np.float32)

            # Mesh
            url = f"airstoreds://{self.assets_table_name}/mesh?frame_id={frame_id}&camera={camera_id}"
            verts = typed.load(url, extension="npbin", dtype=np.float32).reshape(-1, 3)

            # Average texture
            url = f"airstoreds://{self.assets_table_name}/unwrapped_avgtex?frame_id={frame_id}"  # &camera=average"
            avgtex = typed.load(url, extension="png")
            avgtex = einops.rearrange(avgtex, "h w c -> c h w").astype(np.float32)

            # Headpose
            url = f"airstoreds://{self.assets_table_name}/headpose?frame_id={frame_id}&camera={camera_id}"
            headpose = typed.load(url, extension="nptxt", dtype=np.float32)

            if any(i is None for i in (img, verts, avgtex, headpose)):
                raise ValueError(f"Some of fetched data is None for {self.assets_table}-{frame_id}-{camera_id}")

        except Exception as e:
            logger.error(f"Error loading airstore data: {e}")
            return None

        # Background image loading on-the-fly
        bg_img = Image.open(f"{self.bg_model_path}/bg_{camera_id}.png")
        bg_img = bg_img.resize((self.width, self.height))
        bg_img = np.asarray(bg_img)[:, :, :3]  # drop alpha channel
        bg_img = einops.rearrange(bg_img, "h w c -> c h w").astype(
            np.uint8
        )  # bg image is current in 1024x667 resolution

        # pixelcoords
        px, py = np.meshgrid(np.arange(self.width).astype(np.float32), np.arange(self.height).astype(np.float32))
        pixelcoords = np.stack((px, py), axis=-1)

        return dict(
            image=img,
            verts=(verts - self.vertmean) / self.vertstd,
            avgtex=(avgtex - self.texmean) / self.texstd,
            # headpose=headpose,
            frameid=frame_id,
            cameraid=camera_id,
            # id cond info
            neut_verts=(self.neut_vert - self.vertmean) / self.vertstd,
            neut_avgtex=(self.neut_avgtex - self.texmean) / self.texstd,
            # krt info
            campos=np.dot(headpose[:3, :3].T, self.campos[camera_id] - headpose[:3, 3]),
            camrot=np.dot(headpose[:3, :3].T, self.camrot[camera_id].T).T,
            focal=self.focal[camera_id],
            princpt=self.princpt[camera_id],
            modelmatrix=np.eye(4, dtype=np.float32),
            validinput=True,
            pixelcoords=pixelcoords,
            # TODO(julieta) use mask as in previous dataset
            imagemask=np.ones((1, self.height, self.width), dtype=np.float32),
            # ididx and camidx
            idindex=0,
            camindex=self.camera_map[camera_id],  # TODO handle for multi-id
            # bg=bg_img,  # background image  # TODO(julieta) this conflicts
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            frame = self.frames[idx // len(self.frames)]
            camera = self.cameras[idx % len(self.cameras)]
        except IndexError as e:
            print(f"{idx=}, {len(self.frames)=}, {len(self.cameras)=}")
            raise e

        return self.fetch_airstore_data(str(frame), camera)

    def __len__(self):
        return len(self.cameras) * len(self.frames)

    ### Methods added for compat with previous version of dataset. Might want to revisit these
    def get_allcameras(self) -> Set[str]:
        return set(self.cameras)

    def get_img_size(self) -> Tuple[int, int]:
        return (self.height, self.width)
