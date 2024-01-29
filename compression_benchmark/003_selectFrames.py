# Selects 100 files from the MultiFace Dataset for benchmark compression analysis. 
# Randomly selects 2 frames from 5 camera views from 10 subjects, E001_Neutral_Eyes_Open.
import os
import glob
import random

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='selectedIMGFiles/', help='output folder to store compressed files')

args = parser.parse_args()


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
            os.system(f'cp -r {images[j]} {args.output}{naming[1]}+{naming[4]}+{naming[5]}')