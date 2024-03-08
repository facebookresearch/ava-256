import glob
import os
import argparse

parser = argparse.ArgumentParser(description="unzip all zip files in subfolders")
parser.add_argument("--base-dir", default="../datasets/AVA_dataset/")
args = parser.parse_args()

# zip_files = glob.glob(f"{args.base_dir}/*/decoder/head_pose/*.zip")
# # print(id_dirs)

# for zip_file in zip_files:
#     os.system(f'unzip {zip_file} -d {os.path.join(*zip_file.split(os.sep)[:-1])}')


color = glob.glob(f"{args.base_dir}/*/decoder/head_poses/head_pose/")

for c in color:
    os.system(f'mv {c}* {os.path.join(*c.split(os.sep)[:-2])}')