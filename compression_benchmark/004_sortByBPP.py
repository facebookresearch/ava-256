# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os

img_size = 1334 * 2048

img_paths = glob.glob(f"selectedIMGFiles_compressed/AVIF_*/*.AVIF")

total_img_paths = 0
for img_path in img_paths:
    size = os.path.getsize(img_path) * 8  # Get the size in bits.
    x = size / img_size

    if x < 1:
        print(img_path)
        os.symlink(
            "/home/ekim2/Storage/MetaProject/datasets/" + img_path,
            f"selectedIMGFiles_compressed/AVIF-bpp-belowOne/{img_path.split(os.sep)[-2]}+{img_path.split(os.sep)[-1]}",
        )
        total_img_paths += 1
print(total_img_paths)
