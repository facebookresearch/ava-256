# TODO: this gotta be called in the main thread?
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import logging
import os
import io
from typing import Any, Dict, Iterator, Set, Tuple, List, Optional, TypeVar

import cv2
cv2.setNumThreads(0)

import einops
import numpy as np
import torch
import torch as th
from tqdm import tqdm

import torchvision.transforms.v2 as T
from airstore.client.airstore_tabular import AIRStorePathHandler
import fsspec
from care.strict.data.io.file_system.airstore_client import register_airstore_in_fsspec
register_airstore_in_fsspec()
from iopath.common.file_io import PathManager


import care.data.io.typed as typed

from care.strict.data.io.typed.file.obj_np import ObjFileNumpy
from care.strict.data.io.typed.file.image import ImageFile
from care.strict.data.io.typed.file.calib import KRTFile
from care.strict.data.io.typed.file.stdlib import PickleFile

from concurrent.futures import ThreadPoolExecutor
from care.utils.logging import setup_logging
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataloader import default_collate

logger = logging.getLogger(__name__)

MGR_PER_CAPTURE_DATASET: str = "codec_avatar_mgr_12k_per_subject_no_user_data"
MGR_PER_CAPTURE_URL_TEMPLATE: str = "airstoreds://{dataset}/{field}?subject_id={subject_id}"

MGR_PER_FRAME_DATASET: str = "codec_avatar_mgr_12k_frames_combined_no_user_data"
MGR_PER_FRAME_URL_TEMPLATE: str = "airstoreds://{dataset}/{field}?subject_id={subject_id}&segment={segment}&frame_id={frame_id}"

MGR_SEED: int = 42

T_ = TypeVar("T")
def none_collate_fn(items: List[T_]) -> Optional[torch.Tensor]:
    """Modified form of :func:`torch.utils.data.dataloader.default_collate`
    that will strip samples from the batch if they are ``None``."""
    items = [item for item in items if item is not None]
    return default_collate(items) if len(items) > 0 else None


class MultiCaptureDataset(IterableDataset):
    # TODO: add processed data from NFS?

    def __init__(
        self,
        global_rank: int,
        world_size: int,
        per_frame_dataset: str,
        per_capture_dataset: str,
        subjects: List[str],
        limit: int = None,
        n_max_epochs: int = None,
        seed: int = 0,
        prefetch: int = 16,
        shuffle_window: int = 16,
        max_holding_bundles: int = 32,
        s3_server_verbosity_level: int = 0,
    ):
        super().__init__()
        self.path_manager = PathManager()
        self.path_manager.register_handler(AIRStorePathHandler())

        self.subjects = set(subjects)

        self.per_frame_dataset = per_frame_dataset
        self.per_capture_dataset = per_capture_dataset
        self.per_capture_url = MGR_PER_CAPTURE_URL_TEMPLATE

        # airstore opent params
        self.global_rank = global_rank
        self.world_size = world_size
        self.epoch = 0
        self.limit = limit
        self.n_max_epochs = n_max_epochs
        self.seed = seed
        self.prefetch = prefetch
        self.shuffle_window = shuffle_window
        self.max_holding_bundles = max_holding_bundles

        # self.capture = capture
        self.captures = subjects

        downsample = 1
        self.downsample = downsample
        self.height, self.width = (960 // downsample, 720 // downsample)

        # Tuple with all the identifiers of this capture, used in ddp-train
        self.identities = subjects

        # Frames and assets tables
        self.images_table_name = "codec_avatar_mgr_12k_frames_no_user_data"
        self.assets_table_name = "codec_avatar_mgr_12k_assets_v1_no_user_data"
        # Per-capture assets
        self.per_capture_assets = "codec_avatar_mgr_12k_per_subject_no_user_data"

        self.cameras = [0]
        self.texmean = einops.rearrange(np.array([0.485, 0.456, 0.406]) * 255, "c -> c 1 1").astype(np.float32)
        self.texstd  = einops.rearrange(np.array([0.229, 0.224, 0.225]) * 255, "c -> c 1 1").astype(np.float32)
        self.vertmean = np.load("/checkpoint/avatar/julietamartinez/vertmean.npy").astype(np.float32)
        self.vertstd = np.load("/checkpoint/avatar/julietamartinez/vertstd.npy").astype(np.float32).item()

        # Load KRT from NFS
        self.krt, self.focal, self.princpt = dict(), dict(), dict()
        self.neut_vert = dict()
        self.neut_avgtex = dict()

        bad_captures = []
        for capture in tqdm(subjects, desc="Loading per-subject assets"):
            asset_path = f"/checkpoint/avatar/julietamartinez/mgr/assets/{capture}/Segs"

            try:
                self.krt[capture] = typed.load(f"{asset_path}/KRT", extension="krt")
                self.focal[capture] = self.krt[capture]["iPhone_rgb"]["intrin"][:2, :2]
                self.princpt[capture] = self.krt[capture]["iPhone_rgb"]["intrin"][:2,  2]

                # Load neutral data from NFS
                neut_geo = typed.load(f"{asset_path}/neu_geo_v2.obj", extension="obj")
                self.neut_vert[capture] = neut_geo["v"].astype(np.float32)

                neut_tex = typed.load(f"{asset_path}/neu_tex_v2.png", extension="png")
                self.neut_avgtex[capture] = einops.rearrange(neut_tex, "h w c -> c h w").astype(np.float32)
            except Exception as e:
                logger.error(e)
                bad_captures.append(capture)

        if len(bad_captures) > 0:
            logger.warning(f"Bad captures: {bad_captures}")

        self.captures = list(set(self.captures) - set(bad_captures))
        self.captures_set = set(self.captures)
        self.identites = self.captures

        self.subjectid_to_idx = dict()
        for i, capture in enumerate(self.captures):
            self.subjectid_to_idx[capture] = i


    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _open_iterator(self) -> Iterator[Any]:
        # extract numbers of dataloading workers and current worker id (range from
        # 0 to num_workers-1) from torch.utils. If we can't get worker_info we
        # assume the current process is the only dataloading worker.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # split the dataset for each worker
        airstore_world_size = self.world_size * num_workers
        # each worker take it's split by it's parent process rank and worker id
        airstore_rank = self.global_rank * num_workers + worker_id

        return self.path_manager.opent(
            f"airstore://{self.per_frame_dataset}",
            seed=self.epoch + self.seed,
            world_size=airstore_world_size,
            rank=airstore_rank,
            enable_shuffle=True,
            prefetch=self.prefetch,
            shuffle_window=self.shuffle_window,
            max_holding_bundles=self.max_holding_bundles,
            limit=self.limit,
        )

    def num_global_samples(self):
        """Returns the total number of samples in the dataset without sharding."""
        return self.path_manager.opent(
            f"airstore://{self.per_frame_dataset}",
            seed=0,
            world_size=1,  # Will retrieve the entire dataset, not for distributed training
            rank=0,
            enable_shuffle=False,
        ).total_size

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self):
        while self.n_max_epochs is None or self.epoch < self.n_max_epochs:
            logger.info(f"starting epoch: {self.epoch}")
            try:
                for row in self._open_iterator():
                    try:
                        # skipping subjects
                        # if self.subjects is not None and row['subject_id'] not in self.subjects:
                        if self.subjects is not None and row['subject_id'] not in self.captures_set:
                            continue
                        row = self._map_item(row)
                        yield row
                    except Exception as e:
                        logger.warning(f"error {type(e)}:{e} when parsing a row, skipping")
                        raise e
                        continue
                self.epoch += 1
            except StopIteration as e:
                # starting the next epoch
                self.epoch += 1
                if self.epoch >= self.n_max_epochs:
                    raise e
                continue

    def _map_item(self, row) -> Tuple[Any, torch.Tensor]:

        sample = dict()

        sample['image'] = ImageFile.deserialize(row['image'])
        sample['image'] = einops.rearrange(sample['image'], "h w c -> c h w").astype(np.float32)

        fit = PickleFile.deserialize(row['fit'])
        for k in ['rot', 'trans', 'verts', 'torso_verts']:
            sample[k] = fit[k]

        # NOTE(julieta) overwrite verts with torso verts for compat with mugsy
        sample['verts'] = (sample['torso_verts'] - self.vertmean) / self.vertstd
        sample['avgtex'] = einops.rearrange(ImageFile.deserialize(row['tex']), "h w c -> c h w").astype(np.float32)
        sample['avgtex'] = (sample['avgtex'] - self.texmean) / self.texstd

        sample['frameid'] = row['frame_id']
        sample['cameraid'] = 0

        # NOTE(julieta) this will return the id, should convert back to index
        sample['idindex'] = self.subjectid_to_idx[row['subject_id']]
        # Easiert to index below
        capture = row['subject_id']

        sample['neut_verts'] = (self.neut_vert[capture] - self.vertmean) / self.vertstd
        sample['neut_avgtex'] = (self.neut_avgtex[capture] - self.texmean) / self.texstd

        # calib
        sample['focal'] = self.focal[capture]
        sample['princpt'] = self.princpt[capture]

        R, _ = cv2.Rodrigues(sample['rot'])
        sample['camrot'] = R.astype(np.float32)
        sample['campos'] = (-R.T @ sample['trans'][:, None]).astype(np.float32)[:, 0]
        sample['focal'] = np.asarray([sample['focal'][0, 0], sample['focal'][1, 1]], dtype=np.float32)
        sample['princpt'] = sample['princpt'].astype(np.float32)
        sample['modelmatrix'] = np.eye(4, dtype=np.float32)

        sample['validinput'] = True

        # pixelcoords
        px, py = np.meshgrid(np.arange(self.width).astype(np.float32), np.arange(self.height).astype(np.float32))
        sample['pixelcoords'] = np.stack((px, py), axis=-1)
        sample['imagemask'] = np.ones((1, self.height, self.width), dtype=np.float32)

        return sample

    ### Methods added for compat with previous version of dataset. Might want to revisit these
    def get_allcameras(self) -> Set[str]:
        return set([0])

    def get_img_size(self) -> Tuple[int, int]:
        return (self.height, self.width)


def load_frame_lists_raw(subjects: List[str], dataset: str, max_workers: int = 64, url_template: str = None) -> Dict[str, Any]:

    if url_template is None:
        url_template = MGR_PER_CAPTURE_URL_TEMPLATE

    def _load_frame_list(subject_id: str):
        try:
            full_url = url_template.format(dataset=dataset, subject_id=subject_id, field='frame_list'),
            fs, _, parsed = fsspec.core.get_fs_token_paths(full_url)
            data = fs.cat_file(parsed[0])
            data = np.genfromtxt(io.BytesIO(data.tobytes()), dtype=str)
            return data
        except KeyError as key_error:
            logger.warning(f'error when loading {subject_id}')
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        frame_lists = executor.map(_load_frame_list, subjects)

    frame_lists = {
        subject_id : frame_list
        for subject_id, frame_list in zip(subjects, frame_lists) if frame_list is not None
    }
    return frame_lists


def test_iterable_dataset():
    from torch.utils.data import DataLoader

    # using
    mgr_ids_path = '/checkpoint/avatar/timurb/mgr/train_ids_12523.txt'
    mgr_ids = np.loadtxt(mgr_ids_path, dtype=str)[:100]

    dataset = MultiCaptureDataset(
        global_rank=0,
        world_size=1,
        per_frame_dataset=MGR_PER_FRAME_DATASET,
        per_capture_dataset=MGR_PER_CAPTURE_DATASET,
        subjects=mgr_ids,
        n_max_epochs=1,
        seed=MGR_SEED,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
    )

    from PIL import Image

    idx=0
    for batch in loader:
        imgs = batch["image"]

        n, c, h, w = imgs.shape
        for img in imgs:
            img = Image.fromarray(img.numpy().transpose(1, 2, 0).astype(np.uint8))
            img.save(f"/checkpoint/avatar/julietamartinez/samples/mgr/{idx:06d}.png")
            idx += 1

        if idx == 1000:
            break

    # for k, v in batch.items():
    #     if isinstance(v, (th.Tensor, np.ndarray)):
    #         print(f"{k} {v.shape}")


if __name__ == "__main__":
    test_iterable_dataset()
