from PIL import Image
import os
import glob
import pillow_avif
from pillow_heif import register_heif_opener
import time
import json

register_heif_opener()

total_count = 100
img_extension = 'AVIF'

qualities = [0.125, 0.25, 0.5, 0.7, 0.9, 1.0]

input_files = glob.glob('selectedIMGFiles/*.png')
input_files.sort()

for i in range(len(qualities)):
    output_dir = f'selectedIMGFiles_compressed/{img_extension}_100-{qualities[i]}/'
    os.makedirs(output_dir,exist_ok=True)

time_dic = {}

for i in range(len(qualities)):
    output_dir = f'selectedIMGFiles_compressed/{img_extension}_100-{qualities[i]}/'
    cur_count = 0
    
    

    while cur_count < total_count:
        image = Image.open(input_files[cur_count])
        tic = time.time()
        image.save(output_dir+input_files[cur_count].split(os.sep)[-1].split('.')[0] + f'.{img_extension}', quality=int(100*qualities[i]))
        toc = time.time()
        
        time_spent = toc - tic
        fname = input_files[cur_count].split(os.sep)[-1]
        if fname not in time_dic:
            time_dic[fname] = {}
        time_dic[fname][qualities[i]] = time_spent
        cur_count += 1
    
    
with open(f'selectedIMGFiles_compressed/{img_extension}_time-compress.json','w') as f:
    json.dump(time_dic,f)