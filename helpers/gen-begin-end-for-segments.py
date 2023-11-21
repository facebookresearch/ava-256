# take capture signature and segmentation,
# return begin hive row id and end hive row id

# first five columns
# hive row id, mcd, sid, mct, frame_id, camera
# 1376261","20211123","GKH125","1247","63047","401408",

# file naming
# ca_20211123_1247_gkh125_mugsy_fov_full_order_by_frame_sept.csv
# ca_20220202_0844_thn461_mugsy_fov_full_order_by_frame_sept_rowcnt.csv

import argparse
import os
import sys
import numpy as np
import csv
import pandas

path2abplan = '/checkpoint/avatar/jinkyuk/read-only/ablation-plans'

parser = argparse.ArgumentParser(description='sid, mcd, mct for search')
parser.add_argument('--sid', type=str, default=None, help='sid')
parser.add_argument('--mcd', type=str, default=None, help='cmd')
parser.add_argument('--mct', type=str, default=None, help='mct')
#parser.add_argument('--segment', type=str, default=None, help='segment to search frame range')

parsed, unknown = parser.parse_known_args()

for arg in unknown:
    if arg.startswith(("-", "--")):
        parser.add_argument(arg, type=eval)

args = parser.parse_args()

csvdir = '/checkpoint/avatar/jinkyuk/read-only/airstore-dataset-meta/'

fpath = f"{csvdir}/ca_{args.mcd}_{args.mct}_{args.sid}_mugsy_fov_full_order_by_frame_sept.csv"
spath = f"{path2abplan}/{args.mcd}-{args.mct}-{args.sid.lower()}/0/m--{args.mcd}--{args.mct}--{args.sid.upper()}--GHS.seginfo.npy" # sid in filename upper case
seginfo = np.load(spath, allow_pickle=True).tolist()
segments = ['EXP_free_face', 'EXP_jaw003', 'EXP_lip003']


def load_metadata_csv(fpath: str):
    return pandas.read_csv(fpath, header=None)

# Load csv once
metadata_df = load_metadata_csv(fpath)

seg_range_in_hive=dict()

print(args)

for segment in segments:

    assert segment in seginfo.keys()

    if os.path.exists(fpath):
        print(f"{fpath} is found ")
    else:
        print(f"{fpath} is not found ")

    # frames are stored as strs for some reason
    frames = list(map(int, seginfo[segment]))
    print(frames)
    print(" size of frames : {}".format(len(frames)))

    print("line count : {}".format(len(metadata_df)))

    matched_idxs = metadata_df[4].isin(frames)
    rowids = metadata_df[0].loc[matched_idxs].tolist()

    print(" rowids size : {}".format(len(rowids)))
    rowids = sorted(rowids)

    print(" min row id {}, max row id {}".format(rowids[0], rowids[-1]))

    seg_range_in_hive[segment] = (rowids[0], rowids[-1])

    for ii in range(len(rowids)-1):
        if rowids[ii+1] != rowids[ii]+1:
            print(" discontinuous row id found idx {}, rowids[ii] {} rowids[ii+1] {}".format(ii, rowids[ii], rowids[ii+1]))


print(f" RANGE PER SEG :  {seg_range_in_hive}")
dst = f'/checkpoint/avatar/jinkyuk/read-only/airstore-dataset-meta2/{args.mcd}-{args.mct}-{args.sid}-seg-range-in-hive.npy'
np.save(dst, seg_range_in_hive)
