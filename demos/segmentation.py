# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import tempfile

import numpy as np
from PIL import Image, ImageOps
from zipp import Path as ZipPath


def segmentation_demo(ava_dir, subject_id, base_dir, camera_id, frame_id):

    base_dir = f"{ava_dir}/{subject_id}/decoder/"
    path = ZipPath(
        base_dir + "image/" + f"cam{camera_id}.zip",
        f"cam{camera_id}/{int(frame_id):06d}.avif",
    )
    img_bytes = path.read_bytes()
    image = Image.open(io.BytesIO(img_bytes))

    path = ZipPath(
        base_dir + "segmentation_parts/" + f"cam{camera_id}.zip",
        f"cam{camera_id}/{int(frame_id):06d}.png",
    )
    img_bytes = path.read_bytes()
    segmentation = Image.open(io.BytesIO(img_bytes))

    uniques = np.unique(np.array(segmentation))
    segmentation = np.array(segmentation)

    color_map = [
        "#fde725",
        "#c2df23",
        "#86d549",
        "#52c569",
        "#2ab07f",
        "#1e9b8a",
        "#25858e",
        "#2d708e",
        "#38588c",
        "#433e85",
        "#482173",
        "#440154",
    ]

    num_frames = 20

    seg_all = np.zeros((*segmentation.shape, 3))
    alpha = 0

    # os.makedirs("tmp/", exist_ok=True)
    temp_dir = tempfile.TemporaryDirectory()

    for unique, color in zip(uniques, color_map[: len(uniques)]):
        seg = segmentation == unique

        seg = ImageOps.colorize(Image.fromarray(seg).convert("L"), black="black", white=color)
        seg = np.asarray(seg)
        seg_all += seg

    for i in range(num_frames):
        img = np.asarray(seg_all) * alpha + np.asarray(image) * (1 - alpha)
        img = Image.fromarray(np.uint8(img))
        img.save(f"{temp_dir.name}/{i:02d}.png", quality=100)
        alpha += 1 / num_frames

    for i in range(num_frames):
        img = np.asarray(seg_all) * alpha + np.asarray(image) * (1 - alpha)
        img = Image.fromarray(np.uint8(img))
        img.save(f"{temp_dir.name}/{num_frames + i:02d}.png", quality=100)
        alpha -= 1 / num_frames

    os.system(f"ffmpeg -i {temp_dir.name}/%02d.png -r 15 -q:v 3 segmentation_example.apng -y")
    temp_dir.cleanup()
