import io
import os
from pathlib import Path
from typing import Tuple
from zipp import Path as ZipPath
from zipfile import ZipFile

import einops
import numpy as np
import pandas as pd

from PIL import Image
from plyfile import PlyData

import logging


class MugsyCapture:
    """Unique identifier for a Mugsy capture"""

    def __init__(
        self,
        mcd: str,  # Mugsy capture date in 'yyyymmdd' format, eg `20210223`
        mct: str,  # Mugsy capture time in 'hhmm' format, eg `1023`
        sid: str,  # Subject ID, three letters and three numbers, eg `avw368`
        is_relightable: bool = False,  # Whether this is a relightable capture. Ava-256 does not have relightable captures
    ):
        self.mcd = mcd
        self.mct = mct
        self.sid = sid
        self.is_relightable = is_relightable

    def folder_name(self) -> str:
        return f"{self.mcd}--{self.mct}--{self.sid}"


def get_framelist_neuttex_and_neutvert(
    dataset_dir: Path,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load framelist, neutral average texture and neutral vertices
    # TODO(julieta) support zipped and unzipped dataset format
    """

    # Load frame list; ie, (segment, frame) pairs
    frame_list_path = dataset_dir / "frame_list.csv"
    framelist = pd.read_csv(frame_list_path, dtype=str, sep=r",")

    # Neutral conditioning
    neut_framelist = framelist.loc[framelist["seg_id"] == "EXP_neutral_peak"].values.tolist()
    neut_framelist.sort()
    neut_avgtex = None
    neut_vert = None

    avgtex_zip_fname = dataset_dir / "uv_image/color.zip"
    vertices_zip_fname = dataset_dir / "kinematic_tracking/registration_vertices.zip"

    with ZipFile(avgtex_zip_fname, "r") as avgtex_zip, ZipFile(vertices_zip_fname, "r") as vertices_zip:

        for _, neut_frame in neut_framelist:

            verts_path = ZipPath(vertices_zip, at=f"{int(neut_frame):06d}.ply")
            if verts_path.exists():
                ply_bytes = verts_path.read_bytes()
                ply_bytes = io.BytesIO(ply_bytes)
                plydata = PlyData.read(ply_bytes)
                verts = plydata["vertex"].data
                verts = np.array([list(element) for element in verts])
            else:
                logging.info(f"{verts_path} does not exist")
                verts = None

            avgtex_path = ZipPath(avgtex_zip, at=f"color/{int(neut_frame):06d}.avif")
            if avgtex_path.exists():
                img_bytes = avgtex_path.read_bytes()
                img = Image.open(io.BytesIO(img_bytes))
                tex = np.asarray(img)
                tex = einops.rearrange(tex, "h w c -> c h w").astype(np.float32)
            else:
                logging.info(f"{avgtex_path} does not exist")
                tex = None

            # NOTE(julieta) only load one since this might be causing OOM issues
            if tex is not None and verts is not None:
                neut_avgtex = tex
                neut_vert = verts
                break

        if neut_avgtex is None or neut_vert is None:
            raise ValueError("Unable to find any neutral vertices or average textures")

    return framelist, neut_avgtex, neut_vert


def getitem(idx: int, framelist, cameras):
    segment_and_frame = framelist.iloc[idx // len(cameras)]
    segment: str = segment_and_frame.seg_id
    frame: str = segment_and_frame.frame_id
    camera = cameras[idx % len(cameras)]
    return segment, frame, camera
