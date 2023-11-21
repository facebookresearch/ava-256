
# GENERATE MP4 file with sym links and averaged l1 loss value
# python3 process-camab-eval.py --edir [path-to-19 conf exps] --etype [HCAM or TRCAM]

import os
import argparse
import numpy as np
import json 
import cv2

# don't change - all ablation plans keep this info
holdoutcams = ['401036', '401044','401067', '401384', '401412', '401042']  # GHS v2
#traincams=['401045','401315', '401456', '400981', '401404', '400948'] # 401045 are front cam, 5 others are ramdonly selected --> referer to {GHS-manifold-results}/meta/grid.png

parser = argparse.ArgumentParser(description='Train an autoencoder')
parser.add_argument('--edir', type=str, default=None, help='dir for 19 configurations')
parser.add_argument('--savepath', type=str, default=None, help='dir for 19 configurations')

#parser.add_argument('--etype', type=str, default=None, help='TRCAM for TRAIN CAM OR HCAM for HOLDOUT CAM')
args = parser.parse_args()

# edir: directory of testing outputs -- with a specific segment with "TRCAM"traincams or "HCAM"- holdoutcams in the above
edir=args.edir


####### USER ARGUMENT: SET HOLDOUT CAM

cams = holdoutcams

print(f"EDIR : {edir}")

exps=[]

root='/checkpoint/avatar/jinkyuk'
postdir='run-1-nodes-1-gpus-10-dl-workers'

expdir=f"{root}/{edir}/{postdir}"
dirs = os.listdir(expdir)

for dd in dirs:
    jrd = open(f"{expdir}/{dd}/job-config.json", 'rt')
    hratio = json.load(jrd)['holdoutratio']
    exps.append((f"{expdir}/{dd}", hratio))

print(exps)

for e in exps:
    print(e)


h=1024
w=667
repetition=1

for e in exps:
    path=e[0]
    abf=e[1]

    if abf == "0.1" or abf == "1.0":
        print(" PROCESS ABF {}".format(abf))
    else:           
        print(f" SKIP ABF {abf}")
        continue
    
    print(" ABF {}".format(abf))
    print(" PATH {}".format(path))

    #continue

    for cam in cams:
        tmp=f"{args.savepath}/scratch-{abf}-{cam}"
        os.system(f"mkdir -p {tmp}")    
        imgsrc = f"{path}/0/0/image_{cam}"
        print("CAM {} - image symlink {}".format(cam, imgsrc))

        fns = os.listdir(imgsrc)
        fns = sorted(fns)
        
        fidx=0
        for no in range(0, 1000, 10):
            f = f"{no}.jpg"
            spath=os.readlink(f"{imgsrc}/{f}")
            print(" F {}  PATH {} ".format(f, spath.split('/')[-1]))
            toks=spath.split('/')[-1].split('_')
            fr = toks[1]
            cr = toks[2]

            print(" FR {}  CR {}".format(fr, cr))
            assert cr == cam
        
            img = cv2.imread(f"{imgsrc}/{f}")
            g = img[:, w*2:w*3, :]
            g = cv2.putText(g, f"GT", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            r = img[:, w:w*2, :]
            abp = int(float(abf)*100)
            r = cv2.putText(r, f"{abp}% Cameras", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

            for i in range(repetition):
                cv2.imwrite(f'{tmp}/{fidx}.jpg', g)
                fidx += 1
                cv2.imwrite(f'{tmp}/{fidx}.jpg', r)
                fidx += 1
            if fidx > 90:
                break

        #exit()
        #cmd= "ffmpeg -framerate 2 -y -i {}/%d.jpg -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -g 10 -crf 19 {}/ab{}-{}-flip.mp4".format(tmp, args.savepath, abf, cam)
        #os.system(cmd)
        
# make flipping for 1.0  -- 0.5  -- 0.1 -- 0.05 in an image

exit()

for cam in cams:

    if cam == '401042' or cam == '401035':
        continue
    
    sdirs=[]
    #for abf in ['1.0', '0.5', '0.1', '0.05']:
    #for abf in ['1.0', '0.5', '0.1'] : #, '0.05']:
    for abf in ['1.0', '0.1'] : #, '0.05']:                
        tmp=f"{args.savepath}/scratch-{abf}-{cam}"
        sdirs.append(tmp)

    tmpdir = f"{args.savepath}/multiab-{cam}"
    
    os.system(f"mkdir -p {tmpdir}")

    #for i in range(50):
    for i in range(10):    
        imgs=[]
        for sd in sdirs:
            fn = f"{sd}/{i}.jpg"
            print("TRY TO READ {}".format(fn))
            if not os.path.exists(fn):
                print("{} not exist ".format(fn))

            img = cv2.imread(f"{sd}/{i}.jpg")
            imgs.append(img)
        imgout = np.concatenate(imgs, axis=1)
        fn = f"{tmpdir}/{i}.jpg"
        cv2.imwrite(fn, imgout)
        print(fn)
        #break

    cmd= "ffmpeg -framerate 2 -y -i {}/%d.jpg -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -g 10 -crf 19 {}/multiab-cam{}-flip.mp4".format(tmpdir, args.savepath, cam)    
    os.system(cmd)
