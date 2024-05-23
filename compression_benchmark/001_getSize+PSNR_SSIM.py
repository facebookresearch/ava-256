import cv2
import glob
from skimage.metrics import structural_similarity as compare_ssim
from skimage import metrics
from PIL import Image
import numpy
import time
import os
import pillow_avif
from pillow_heif import register_heif_opener
import json
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

# scales = [12, 25, 50, 70, 90, 100]

images = glob.glob(f"{args.output}{args.img_extension}_100-100/*")
images.sort()
imgs_png = glob.glob(f"{args.dir}*")
imgs_png.sort()

ssim_dic = {}
psnr_dic = {}
decoding_time_dic = {}

for image, img_png in zip(images, imgs_png):
    original_img_png = Image.open(img_png).convert("RGB")

    ssim_list = {}
    psnr_list = {}
    time_list = {}
    for scale in scales:
        tic = time.time()
        img_scaled = Image.open(
            args.output + image.split(os.sep)[1][:-3] + str(scale) + "/" + image.split(os.sep)[-1]
        ).convert("RGB")
        toc = time.time()

        ssim, _ = compare_ssim(numpy.array(original_img_png), numpy.array(img_scaled), full=True, multichannel=True)
        psnr = metrics.peak_signal_noise_ratio(numpy.array(original_img_png), numpy.array(img_scaled))

        ssim_list[scale] = ssim
        psnr_list[scale] = psnr
        time_list[scale] = toc - tic

    ssim_dic[image.split(os.sep)[-1]] = ssim_list
    psnr_dic[image.split(os.sep)[-1]] = psnr_list
    decoding_time_dic[image.split(os.sep)[-1]] = time_list

with open(f"{args.output}{args.img_extension}_ssim.json", "w") as f:
    json.dump(ssim_dic, f)

with open(f"{args.output}{args.img_extension}_psnr.json", "w") as f:
    json.dump(psnr_dic, f)

with open(f"{args.output}{args.img_extension}_time-decompress.json", "w") as f:
    json.dump(decoding_time_dic, f)
