# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import pillow_avif
from PIL import Image
from pillow_heif import register_heif_opener

from utils import *

register_heif_opener()


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", default="selectedIMGFiles/*.png", help="folder with non-lossy png image files")
parser.add_argument(
    "-o", "--output", default="selectedIMGFiles_compressed/", help="output folder to store compressed files"
)
parser.add_argument(
    "-e", "--img_extension", default="jpg", help="extension format of the compressed file (i.e. jpg, avif, heic, webp)"
)

args = parser.parse_args()

input_files = glob.glob(args.dir)
input_files.sort()

for qual in qualities:
    output_dir = Path(f"{args.output}{args.img_extension}_100-{qual}/")
    output_dir.mkdir(parents=True, exist_ok=True)

encoding_time_dic = defaultdict(lambda: {})

for qual in qualities:
    output_dir = f"{args.output}{args.img_extension}_100-{qual}/"
    cur_count = 0

    while cur_count < total_image_count:
        image = Image.open(input_files[cur_count])
        tic = time.time()
        image.save(
            output_dir + input_files[cur_count].split(os.sep)[-1].split(".")[0] + f".{args.img_extension}", quality=qual
        )
        toc = time.time()

        time_spent = toc - tic
        basename = input_files[cur_count].split(os.sep)[-1][:-3] + "png"
        encoding_time_dic[basename][qual] = time_spent
        cur_count += 1


with open(f"{args.output}{args.img_extension}_time-compress.json", "w") as f:
    json.dump(encoding_time_dic, f)
