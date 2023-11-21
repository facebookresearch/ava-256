

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

path2abplan = '/checkpoint/avatar/jinkyuk/read-only/ablation-plans'

parser = argparse.ArgumentParser(description='sid, mcd, mct for search')
parser.add_argument('--sid', type=str, default=None, help='sid')
parser.add_argument('--mcd', type=str, default=None, help='cmd')
parser.add_argument('--mct', type=str, default=None, help='mct')
parser.add_argument('--segment', type=str, default=None, help='segment to search frame range')

parsed, unknown = parser.parse_known_args()
for arg in unknown:
    if arg.startswith(("-", "--")):
        parser.add_argument(arg, type=eval)
        
args = parser.parse_args()

csvdir = '/checkpoint/avatar/jinkyuk/read-only/airstore-dataset-meta/'

fpath = f"{csvdir}/ca_{args.mcd}_{args.mct}_{args.sid}_mugsy_fov_full_order_by_frame_sept.csv"

spath = f"{path2abplan}/{args.mcd}-{args.mct}-{args.sid.lower()}/0/m--{args.mcd}--{args.mct}--{args.sid.upper()}--GHS.seginfo.npy" # sid in filename upper case

seginfo = np.load(spath, allow_pickle=True).tolist()

assert args.segment in seginfo.keys()

print(args)

if os.path.exists(fpath):
    print(f"{fpath} is found ")    
else:
    print(f"{fpath} is not found ")


frames = seginfo[args.segment]
print(frames)
print(" size of frames : {}".format(len(frames)))

os.system(f'wc -l {fpath}')

#exit()

#2275969

rowids = []
with open(fpath, 'rt') as rfd:

    lcnt = 0 
    line = rfd.readline()
    while line:    
        lcnt += 1

        tokens = line.split(',')
        rowid = tokens[0].replace("\"", "")
        frameid = tokens[4].replace("\"", "")
        
        if lcnt < 10 or lcnt > (2275969-10):
            print(" lcnt : {}, rowid {}, frameid {}".format(lcnt, rowid, frameid))


        if "{:06}".format(int(frameid)) in frames:
            #print(f" find a sample row that belongs to segnemt {args.segment}'s frame list ")
            rowids.append(rowid)
            
        line = rfd.readline()





print("line count : {}".format(lcnt))

print(" rowids size : {}".format(len(rowids)))
rowids = sorted(rowids)

print(" min row id {}, max row id {}".format(rowids[0], rowids[-1]))

for ii in range(len(rowids)-1):
    if int(rowids[ii+1]) != int(rowids[ii])+1:
        print(" discontinuous row id found idx {}, rowids[ii] {} rowids[ii+1] {}".format(ii, rowids[ii], rowids[ii+1]))
