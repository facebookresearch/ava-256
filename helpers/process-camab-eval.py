
# GENERATE MP4 file with sym links and averaged l1 loss value
# python3 process-camab-eval.py --edir [path-to-19 conf exps] --etype [HCAM or TRCAM]

import os
import argparse
# don't change - all ablation plans keep this info
holdoutcams = ['401036', '401044','401067', '401384', '401412', '401042']  # GHS v2
traincams=['401045','401315', '401456', '400981', '401404', '400948'] # 401045 are front cam, 5 others are ramdonly selected --> referer to {GHS-manifold-results}/meta/grid.png

parser = argparse.ArgumentParser(description='Train an autoencoder')
parser.add_argument('--edir', type=str, default=None, help='dir for 19 configurations')
parser.add_argument('--etype', type=str, default=None, help='TRCAM for TRAIN CAM OR HCAM for HOLDOUT CAM')
args = parser.parse_args()

# edir: directory of testing outputs -- with a specific segment with "TRCAM"traincams or "HCAM"- holdoutcams in the above
edir=args.edir

####### USER ARGUMENT: SET HOLDOUT CAM
if args.etype == "HCAM":
    cams = holdoutcams
elif args.etype == "TRCAM":
    cams = traincams
else:
    assert(False)

print(f"EDIR : {edir}")

exps=[]

root='/checkpoint/avatar/jinkyuk'
postdir='run-1-nodes-8-gpus-4-dl-workers'

expdir=f"{root}/{edir}/{postdir}"
dirs = os.listdir(expdir)

for dd in dirs:
    d = f"{expdir}/{dd}/0/0"
    files = os.listdir(d)
    pngcnt=0
    for e in files:
        if "png" in e:
            pngcnt += 1
    if pngcnt > 2 :
        exps.append(f"{expdir}/{dd}")
print(exps)

for i, e in enumerate(exps):
    print(f"{i} - {e}")

#exit()

# imagefile list per cam
imgs={}

# loss value file list per cam
loss={}

# TODO COLLECT

gimgs={}
gloss={}

gframes=dict()

for ie, e in enumerate(exps):
    imgs = dict()
    loss = dict()

    for cam in cams:
        imgs[cam] = []
        loss[cam] = []

    dpath = f"{e}/0/0/"
    files = os.listdir(dpath) # assume that listdir reutrn sorted file list
    for fn in files:
        if "img" in fn:
            #img_0_078766_401042.png
            cam = fn.split('_')[3].split('.')[0]
            imgs[cam].append(fn)

        #elif "l1" in fn and "fmask" not in fn: # l1 loss without facemask l1_{subjectid}_{frameid}_{cam}.txt"
        elif "l1" in fn and "fmask" in fn: # l1 loss without facemask l1_{subjectid}_{frameid}_{cam}_fmask.txt"
            #l1_0_078419_401067_fmask.txt
            cam = fn.split('_')[3].split('.')[0]
            loss[cam].append(fn)
            frameid= fn.split('_')[2]
            if frameid not in gframes.keys():
                gframes[frameid] = 1
            else:
                gframes[frameid] += 1

    for k, v in imgs.items():
        v = sorted(v)
        print(f"IMG {k} -- {len(v)} -- {v}")
    for k, v in loss.items():
        v = sorted(v)
        print(f"LOSS {k} -- {len(v)} -- {v}")

    gimgs[e] = imgs
    gloss[e] = loss

commonframes=[]
for k, v in gframes.items():
    if v == len(cams)*len(exps):
        commonframes.append(k)

print(f"@@@@@@@@ Common frames length :{len(commonframes)} - {commonframes}")

for e in exps:
    imgs = gimgs[e]
    loss = gloss[e]
    newloss = {}
    for cam, v in loss.items():
        newv = list()
        for fn in v:
            frameid= fn.split('_')[2]
            if frameid in commonframes:
                newv.append(fn)
            else:

                print(f" cam {cam} frame {fn} not not in common frame ")
        newloss[cam] = newv
    gloss[e] = newloss

#exit()

for ie, e in enumerate(exps):
    dpath = f"{e}/0/0/"

    #MP4 file generation
    #step1: generate sym link
    #make symbolic link
    print("images : @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    for k, v in imgs.items():
        print(" creating sym link for {} cam ".format(k))
        symdir = f"{dpath}/image_{k}"
        os.system(f"mkdir {symdir}")
        for i, fn in enumerate(v):
            cmd=f"ln -s {dpath}/{fn} {symdir}/{i}.png"
            print(cmd)
            os.system(cmd)
    # make mp4 file with sym link
    print("merging images : @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    for k, v in imgs.items():
        print(" creating mp4 file for {} cam ".format(k))
        symdir = f"{dpath}/image_{k}"
        cmd= "ffmpeg -framerate 30 -y -i {}/%d.png -g 10 -crf 19 {}/{}-merged.mp4".format(symdir, dpath, k)
        print(cmd)
        os.system(cmd)

    # # make loss value list
    print("loss: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    loss = gloss[e]
    for k, v in loss.items():
        print(f" process {k} cam -- len v : {len(v)}")
        #print(f" cam {k}: {v}")
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        sdir = f"{dpath}/loss_{k}"
        os.system(f"mkdir {sdir}")
        wfd=open(f"{sdir}/l1.txt", "wt")
        wfd2=open(f"{sdir}/l1-fr-loss.txt", "wt")
        singlerow = ""
        for i, fn in enumerate(v):
            frameid = fn.split('_')[2]
            if frameid != commonframes[i]:
                print(f" MISMATCH in frame id and position -- fn's frameid{frameid} -- commonframe[{i}]: {commonframes[i]}")
            val=open(f"{dpath}{fn}").readlines()[0].rstrip()
            singlerow += f"{val}, "
            wfd2.write(f"{frameid}:{val}, ")
        wfd.write(singlerow)
        wfd.close()
        wfd2.close()
