import logging
import os

from io import StringIO

from PIL import Image

from care.data.io.typed.mugsy_raw import (
    CamByteTransform,
    load_zip_info_synced,
    PIXEL_FORMAT_MAPPING,
    ROTATE_FORMAT_MAPPING,
)
from care.strict.data.io.typed.file.calib import KRTFile
from care.strict.data.io.typed.file.stdlib import TextFile

from typing import Any, Dict, List, Tuple, Literal, Callable, Optional, Sequence


def find_sync_file(
    metadata_path: str,
    mcd: str,
    mct: str,
    sid: str,
    camera: str,
) -> Optional[str]:
    return os.path.join(metadata_path, f"m--{mcd}--{mct}--{sid.upper()}--GHS-{camera}.zip.info.synced")


def find_calib_file(metadata_path) -> Optional[str]:
    for (dirpath, dirnames, filenames) in os.walk(metadata_path):
        for filename in filenames:
            if filename.endswith('KRT2'):
                return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"No KRT2 file found in {metadata_path}")


def process_sync_file(sync_file) -> Dict:
    if sync_file is not None:
        try:
            value = TextFile.load(sync_file)
            segments, timestamps, properties = load_zip_info_synced(StringIO(value))
            sync_file_dict = {
                "segments": segments,
                "timestamps": timestamps,
                "properties": properties,
            }
        except FileNotFoundError as e:
            # raise FileNotFoundError(
            #     f"Unable to load synchronization information from {sync_file}"
            # ) from e
            print(f"Unable to load synchronization information from {sync_file}")
            return {}

        print(f"Found sync file {sync_file}")
    if "timestamps" not in sync_file_dict or "properties" not in sync_file_dict:
        raise ValueError("Missing required information from the sync file.")

    return sync_file_dict


def process_sync_files(
    metadata_path: str,
    mcd: str,
    mct: str,
    sid: str,
    camera_list: Sequence[str],
):
    sync_file_metadata = {}
    for camera in camera_list:
        sync_file = find_sync_file(metadata_path, mcd, mct, sid, camera)
        if camera not in sync_file_metadata:
            sync_file_dict = process_sync_file(sync_file)
            if sync_file_dict:
                sync_file_metadata[camera] = sync_file_dict
                # Otherwise add nothing I guess

    return sync_file_metadata


def process_calib_file(calib_file):
    if calib_file is not None:
        try:
            krt_dict = KRTFile.load(calib_file)
        except FileNotFoundError as e:
            print(f"Found calibration file {calib_file}")
            raise FileNotFoundError(
                f"Unable to load calibration information from {calib_file}"
            ) from e
    return krt_dict


def get_camera_calibration_data(calib_file_dict, camera):
    try:
        camera_calib_data = calib_file_dict[camera]
    except KeyError as e:
        raise ValueError(
            f"Unable to find camera {camera} from the loaded calibration file {calib_file_parsed}."
        ) from e
    # Check calibration info if undistort is set to true.
    if "intrin" not in camera_calib_data or "dist" not in camera_calib_data:
        raise ValueError(
            "Calibration file must contain both camera intrinsic and distortion coefficient."
        )
    if "model" in camera_calib_data and camera_calib_data["model"] != "radial-tangential":
        raise NotImplementedError(
            '"radial-tangential" is the only supported distortion model.'
        )
    return camera_calib_data


def parse_calib_properties(properties):
    # Camera rotation
    rotation = int(properties.get("Rotation", -1))
    cv_rotate_format = ROTATE_FORMAT_MAPPING[rotation]

    # Must have these properties. Won't work on very old sequences.
    # Pixel pattern
    pixel_format = properties["PixelFormat"]
    if pixel_format not in PIXEL_FORMAT_MAPPING:
        raise ValueError(f"Unknown pixel format: {pixel_format}")
    cv_pixel_format = PIXEL_FORMAT_MAPPING[pixel_format]

    # Must have these properties. Won't work on very old sequences.
    return {
        "height": int(properties["Height"]),
        "width": int(properties["Width"]),
        "cv_rotate_format": cv_rotate_format,
        "cv_pixel_format": cv_pixel_format,
        # compression method, default to bz2
        "compression_method": str(properties.get("Compression", "bz2")),
        # Must have fixed WB. Won't work pre- mid-2018 sequences
        # White balancing
        "manual_white_balance": int(properties.get("ManualWB", 0)) != 0,
    }

def generate_transform(dataset_metadata: Dict, camera: str, downsample_factor=None):
    if dataset_metadata is None or 'krt_dict' not in dataset_metadata or 'sync_file_metadata' not in dataset_metadata:
        raise ValueError(f"invalid metadata")

    calib_file_dict = dataset_metadata['krt_dict']
    sync_file_metadata = dataset_metadata['sync_file_metadata']

    if camera not in calib_file_dict or camera not in sync_file_metadata:
        # raise ValueError(f"camera metadata missing for camera {camera}")
        print(f"camera metadata missing for camera {camera}")
        return

    sync_file_dict = sync_file_metadata[camera]
    parsed_properties = parse_calib_properties(properties=sync_file_dict["properties"])

    camera_calib_data = get_camera_calibration_data(calib_file_dict, camera)

    # Load the KRT2 file to get intrinsics and distortion if needed.
    K = camera_calib_data["intrin"]
    dist = camera_calib_data["dist"]
    model = camera_calib_data["model"]

    transform = CamByteTransform(
        white_balance=True,
        debayer=True,
        rotate=True,
        undistort=True,
        prioritize_downsample=True,
        down_sample_factor=downsample_factor,
        K=K,
        distortion=dist,
        distortion_model=model,
        reduce_memory_footprint=True,
        **parsed_properties,
    )
    return transform

def process_tile_png_sample(transform, sample, sample_id=0, log_image=False):
    transform_img = transform(memoryview(sample['frame']))

    if log_image:
        rgb_img = Image.fromarray(transform_img, 'RGB')
        logging.info(f"$$_sample_{sample_id}_frame_{frame_id}_camera_{camera}")
        rgb_img.save(f"{sample_id}_frame_{frame_id}_camera_{camera}_rgb.png")

    return transform_img


def process_dataset_metadata(
    metadata_path: str,
    mcd: str,
    mct: str,
    sid: str,
) -> Dict:
    """Load calibration and synchronization files"""
    calib_file = find_calib_file(metadata_path)
    calib_file_dict = process_calib_file(calib_file)
    cameras = list(calib_file_dict.keys())

    # TODO(julieta) bring this back to normal once the sync info issue is resolved
    # sync_file_metadata = process_sync_files("/uca/julieta/m--20210223--1023--AVW368--GHS/minisis", mcd, mct, sid, cameras)
    sync_file_metadata = process_sync_files(metadata_path, mcd, mct, sid, cameras)

    dataset_metadata =  {
        "krt_dict" : calib_file_dict,
        "sync_file_metadata" : sync_file_metadata,
    }
    return dataset_metadata
