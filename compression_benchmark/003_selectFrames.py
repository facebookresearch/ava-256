import os
import glob
import random

folders = glob.glob('multiface_subset/*/')
folders.sort()

for f in folders:
    cams = glob.glob(f'{f}images/E001_Neutral_Eyes_Open/*/')
    random.shuffle(cams)
    for i in range(5):
        images = glob.glob(f'{cams[i]}*.png')
        random.shuffle(images)
        for j in range(2):
            naming = images[j].split(os.sep)
            os.system(f'cp -r {images[j]} selectedIMGFiles/{naming[1]}+{naming[4]}+{naming[5]}')