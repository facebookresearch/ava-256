from PIL import Image
import os
import glob
import pillow_avif
from pillow_heif import register_heif_opener
import time
import json
import argparse

register_heif_opener()


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', default='selectedIMGFiles/*.png', help='folder with non-lossy png image files')
parser.add_argument('-o', '--output', default='selectedIMGFiles_compressed/', help='output folder to store compressed files')
parser.add_argument('-e', '--img_extension', default='jpg', help='extension format of the compressed file (i.e. jpg, avif, heic, webp)')

args = parser.parse_args()

total_count = 100
# img_extension = 'AVIF'

qualities = [12, 25, 50, 70, 90, 100]

input_files = glob.glob(args.dir)
input_files.sort()

for i in range(len(qualities)):
    output_dir = f'{args.output}{args.img_extension}_100-{qualities[i]}/'
    os.makedirs(output_dir,exist_ok=True)

time_dic = {}

for i in range(len(qualities)):
    output_dir = f'{args.output}{args.img_extension}_100-{qualities[i]}/'
    cur_count = 0
    
    

    while cur_count < total_count:
        image = Image.open(input_files[cur_count])
        tic = time.time()
        image.save(output_dir+input_files[cur_count].split(os.sep)[-1].split('.')[0] + f'.{args.img_extension}', quality=qualities[i])
        toc = time.time()
        
        time_spent = toc - tic
        fname = input_files[cur_count].split(os.sep)[-1]
        if fname not in time_dic:
            time_dic[fname] = {}
        time_dic[fname][qualities[i]] = time_spent
        cur_count += 1
    
    
with open(f'{args.output}{args.img_extension}_time-compress.json','w') as f:
    json.dump(time_dic,f)