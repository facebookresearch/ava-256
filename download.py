# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset download
"""

import argparse
import json
import logging
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple, Union
from urllib.error import HTTPError, URLError

import pandas as pd

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

BPATH = "https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/ava-256/"

# Assets that we can download
ASSETS: Dict[str, List[str]] = OrderedDict(
    {
        "all": [],  # This just means we download all assets
        "camera_calibration": ["camera_calibration.json"],
        "frame_list": ["frame_list.csv"],
        "head_pose": ["head_pose/head_pose.zip"],
        "image": ["image/cam{camera}.zip"],
        "keypoints_3d": ["keypoints_3d/keypoints_3d.zip"],
        "kinematic_tracking": [
            "kinematic_tracking/registration_vertices_mean.npy",
            "kinematic_tracking/registration_vertices_variance.txt",
            "kinematic_tracking/registration_vertices.zip",
        ],
        # NOTE(julieta) no light info in ava256
        # "lights": [
        #     "lights/lights_light_calibration.txt",
        #     "lights/light_pattern.txt",
        # ],
        "segmentation_parts": ["segmentation_parts/cam{camera}.zip"],
        "uv_image": [
            "uv_image/color_mean.png",
            "uv_image/color_variance.txt",
            "uv_image/color.zip",
        ],
    }
)

MULTI_CAM_ASSETS = ["image", "segmentation_parts"]


@dataclass(frozen=True)
class MugsyCapture:
    """Unique identifier for a Mugsy capture"""

    mcd: str  # Mugsy capture date in 'yyyymmdd' format, eg `20210223`
    mct: str  # Mugsy capture time in 'hhmm' format, eg `1023`
    sid: str  # Subject ID, three letters and three numbers, eg `avw368`
    is_relightable: bool = False  # Whether this is a relightable capture. Ava-256 does not have relightable captures

    def folder_name(self) -> str:
        return f"{self.mcd}--{self.mct}--{self.sid}"


def load_captures(captures_path: Union[str, Path]) -> List[MugsyCapture]:
    captures = pd.read_csv(captures_path, dtype=str)
    captures = [
        MugsyCapture(mcd=row["mcd"], mct=row["mct"], sid=row["sid"], is_relightable=True)
        for _, row in captures.iterrows()
    ]
    return captures


def get_camera_list(bpath: str, capture: MugsyCapture) -> List[str]:
    url = bpath + f"{capture.folder_name()}" + "/decoder/" + ASSETS["camera_calibration"][0]
    try:
        with urllib.request.urlopen(url) as response:
            cameras_dict = json.load(response)
    except HTTPError as e:
        logging.error("HTTP error occurred reaching %s: %s", url, e)

    camera_list = [x["cameraId"] for x in cameras_dict["KRT"]]
    return sorted(camera_list)


# def download_link(from_url: str, to_path: str) -> None:
def download_link(from_url_and_to_path: Tuple[int, int, str, str]) -> None:
    """Download a single link"""
    i, total, from_url, to_path = from_url_and_to_path

    to_path.parent.mkdir(parents=True, exist_ok=True)

    if to_path.exists():
        # TODO(julieta) Local file could be corrupted or outdated, check hashes instead of just skipping.
        logging.info("%s already exists, skipping", to_path)
        return

    percent_done = 100 * i / total
    logging.info("[%.2f%%] Downloading link %d / %d from %s to %s", percent_done, i, total, from_url, to_path)

    try:
        urllib.request.urlretrieve(from_url, to_path)
    except HTTPError as e:
        logging.error("HTTP error occurred reaching %s: %s", from_url, e)
        raise e
    except URLError as e:
        logging.error("URL error occurred reaching %s: %s", from_url, e)
        raise e


def download_links(links_and_paths: List[Tuple[int, int, str, str]]) -> None:
    """Download a bunch of links to a series of filesystem paths"""
    for i, total, from_url, to_path in links_and_paths:
        download_link((i, total, from_url, to_path))


def main():
    parser = argparse.ArgumentParser(description="Download the ava-256 dataset")
    parser.add_argument("--output-dir", "-o", type=str, help=f"Directory to write the dataset to", required=True)
    parser.add_argument("--captures-file", type=str, default="256_ids.csv", help="CSV file with captures to download")
    parser.add_argument(
        "--assets", type=str, default=["all"], nargs="+", help=f"List of assets to download. Must be in {ASSETS.keys()}"
    )
    parser.add_argument("-n", type=int, default=16, help="Number of captures from captures-file download")
    parser.add_argument("--workers", "-j", type=int, default=8, help="Number of workers for parallel download")
    parser.add_argument("--size", "-s", type=str, default="4TB", choices=["4TB", "8TB", "16TB", "32TB"])
    # TODO(julieta) let people pass a single sid
    # TODO(julieta) check the hash of the remote files and compare with local files
    args = parser.parse_args()

    # TODO(julieta) check version match, if mismatch, then delete/download new data

    # Check assets flag
    if "all" in args.assets and len(args.assets) > 1:
        raise ValueError("Cannot use 'all' together with other assets")

    for asset in args.assets:
        if asset not in ASSETS.keys():
            raise ValueError(f"Invalid asset '{asset}'. Must be one of {ASSETS.keys()}")

    if "all" in args.assets and len(args.assets) == 1:
        args.assets = ASSETS
        args.assets.pop("all")
        logging.info("Downloading all assets")
    else:
        args.assets = {x: ASSETS[x] for x in args.assets}

    # Check captures file
    captures = load_captures(args.captures_file)
    if args.n > len(captures):
        raise ValueError(f"Requested more captures ({args.n}) than available in captures file ({len(captures)})")
    logging.info(
        "Downloading the first %d out of %d captures from %s",
        args.n,
        len(captures),
        args.captures_file,
    )
    captures = captures[: args.n]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    camera_lists_dict = dict()
    links_and_paths: List[Tuple] = []
    dataset_path = BPATH + f"{args.size}/"

    # TODO(julieta) check that these are valid captures, sid mcd and mct are there
    for capture in captures:
        capture_path = capture.folder_name()
        logging.info("Working on capture %s", capture_path)

        for assets, asset_paths in args.assets.items():

            if assets in MULTI_CAM_ASSETS:

                # Make sure we have the list of cameras for this capture
                if capture not in camera_lists_dict:
                    camera_list = get_camera_list(dataset_path, capture)
                    camera_lists_dict[capture] = camera_list

                    logging.info(
                        "Found %s cameras for %s: %s",
                        len(camera_list),
                        capture,
                        camera_list,
                    )

                camera_list = camera_lists_dict[capture]

                # Generate download links for all the cameras
                for camera in camera_list:
                    asset_path = asset_paths[0].format(camera=camera)
                    from_url = dataset_path + capture_path + "/decoder/" + asset_path
                    to_path = output_dir / capture_path / "decoder" / asset_path
                    links_and_paths.append((from_url, to_path))

            else:

                # Generate donwload links for all the assets
                for asset_path in asset_paths:
                    from_url = dataset_path + capture_path + "/decoder/" + asset_path
                    to_path = output_dir / capture_path / "decoder" / asset_path
                    links_and_paths.append((from_url, to_path))

    # Done creating links, donwload everything
    total_links = len(links_and_paths)
    links_and_paths = [(i + 1, total_links, link, path) for i, (link, path) in enumerate(links_and_paths)]

    n_workers = min(args.workers, cpu_count())
    logging.info("Downloading %s files with %s workers", len(links_and_paths), n_workers)

    if n_workers == 1:
        logging.warning("Downloading with a single worker. This might be slow, consider using more workers.")
        download_links(links_and_paths)
    else:
        pool = Pool(n_workers)
        download_func = partial(download_link)
        pool.imap(download_func, links_and_paths)
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
