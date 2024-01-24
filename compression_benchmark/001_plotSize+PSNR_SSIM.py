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

register_heif_opener()

scales = [0.125, 0.25, 0.5, 0.7, 0.9, 1.0]

met = ['SSIM','PSNR']
extension = 'webp'
images = glob.glob(f'selectedIMGFiles_compressed/{extension}_100-1.0/*')
images.sort()
imgs_png = glob.glob(f'selectedIMGFiles/*')
imgs_png.sort()

cumulative_scores = numpy.zeros(len(scales))
cumulative_times = numpy.zeros(len(scales))

ssim_dic = {}
psnr_dic = {}
time_dic = {}

for image,img_png in zip(images,imgs_png):
    original_img_png = Image.open(img_png).convert('RGB')

    ssim_list = {}
    psnr_list = {}
    time_list = {}
    for scale in scales:      
        tic = time.time()
        img_scaled = Image.open('selectedIMGFiles_compressed/' + image.split(os.sep)[1][:-3] + str(scale) + '/' + image.split(os.sep)[-1]).convert('RGB')
        toc = time.time()
        
        ssim, _ = compare_ssim(numpy.array(original_img_png), numpy.array(img_scaled), full=True, multichannel=True)
        psnr = metrics.peak_signal_noise_ratio(numpy.array(original_img_png), numpy.array(img_scaled))
        
        ssim_list[scale] = ssim
        psnr_list[scale] = psnr
        time_list[scale] = toc - tic
        
    ssim_dic[image.split(os.sep)[-1]] = ssim_list
    psnr_dic[image.split(os.sep)[-1]] = psnr_list
    time_dic[image.split(os.sep)[-1]] = time_list

with open(f'selectedIMGFiles_compressed/{extension}_ssim.json','w') as f:
    json.dump(ssim_dic,f)
    
with open(f'selectedIMGFiles_compressed/{extension}_psnr.json','w') as f:
    json.dump(psnr_dic,f)

with open(f'selectedIMGFiles_compressed/{extension}_time-decompress.json','w') as f:
    json.dump(time_dic,f)