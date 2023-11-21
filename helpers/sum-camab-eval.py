
# python3 sum-camab-eval.py --edir [path to 19 exps] --etype [HCAM or TRCAM] --savepath [path to save outputs]

import copy
import os
import argparse
# don't change - all ablation plans keep this info
holdoutcams = ['401036', '401044','401067', '401384', '401412', '401042']  # GHS v2
traincams=['401045','401315', '401456', '400981', '401404', '400948'] # 401045 are front cam, 5 others are ramdonly selected --> referer to {GHS-manifold-results}/meta/grid.png

parser = argparse.ArgumentParser(description='Train an autoencoder')
parser.add_argument('--edir', type=str, default=None, help='dir for 19 configurations')
parser.add_argument('--etype', type=str, default=None, help='TRCAM for TRAIN CAM OR HCAM for HOLDOUT CAM')
parser.add_argument('--savepath', type=str, default=None, help='TRCAM for TRAIN CAM OR HCAM for HOLDOUT CAM')
args = parser.parse_args()

# edir: directory of testing outputs -- with a specific segment with "TRCAM"traincams or "HCAM"- holdoutcams in the above

edir=args.edir # @@@@@@ WHERE 19 exps with 19 different ab factors are saved

####### USER ARGUMENT: SET HOLDOUT CAM
if args.etype == "HCAM":
    cams = holdoutcams
elif args.etype == "TRCAM":
    cams = traincams
else:
    assert(False)

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
#print(exps)

print(f"@@@@ EDIR : {edir} -- LENGTHL {len(exps)}")

summaryfile = f"{args.savepath}/{edir}.summary.txt"
ewfd = open(summaryfile, 'wt')

sumdict=dict() # cam : dict of (hratio: line)
for cam in cams:
    e19 = dict()
    #for h in ['1.0', '0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1', '0.09', '0.08', '0.07', '0.06', '0.05', '0.04', '0.03', '0.02', '0.01']:
    #    e19[h] = None
    sumdict[cam] = copy.deepcopy(e19)

for ie, e in enumerate(exps):
    dpath = f"{e}/0/0/"

    files=os.listdir(e)
    ofile = None
    for fn in files:
        if ".out" in fn and "sbatch" in fn:
            ofile = fn
            break
    #print(ofile)
    olines = open(f"{e}/{ofile}", 'rt').readlines()
    econfline = None
    for ee in olines:
        if 'Namespace' in ee and 'INFO' in ee and 'args.rank' in ee and 'root' in ee:
            econfline = ee
            break
    assert(econfline)
    toks = econfline.split()
    #print(toks)
    #for i, tok in enumerate(toks):
    #    print(f"\t {i} -- {tok}")
    assert(toks[22].split('=')[0] == 'holdoutratio')
    hratio = toks[22].split('=')[1].replace("'", "").replace(',', '')

    for k in cams:
        print(f"@@@@@@ IDX:{ie} EXP:{e} abfactor:{hratio} CAM:{k}")

        #MP4 file nameing: {cam}-merged.mp3 - exmple: 401412-merged.mp4
        mp4=f"{dpath}/{k}-merged.mp4"
        mp4dst = f"{args.savepath}/{edir}{k}_{hratio}_merged.mp4"
        mp4cmd =f"cp {mp4} {mp4dst}"
        print(f"{mp4cmd}")
        os.system(mp4cmd)
        #continue

        sdir = f"{dpath}/loss_{k}"
        line=open(f"{sdir}/l1.txt", "rt").readlines()
        #print(line)
        e = e.replace('/', '+').replace('-', '+')

        wfile = f"{args.savepath}/{edir}_{k}_{hratio}.txt"
        print(f"writing to {wfile}")
        wfd = open(wfile, 'wt')
        assert(len(line) == 1)
        wfd.write(line[0])
        wfd.close()
        #summary=f"{edir}_{hratio}_{k}, "+line[0]+"\n"
        summary=f"ab_{hratio}_{k}, "+line[0]+"\n"



        #ewfd.write(f"{edir}_{hratio}_{k}, "+line[0]+"\n")
        sumdict[k][hratio] = summary

#exit()

#ewfd.close()
print(f"@@@@@@@@ {edir} summary saved to {summaryfile}")

# sumdict=dict() # cam : dict of (hratio: line)
for k, v in sumdict.items():
    print(f"@@@@@@CAMERA : {k} -- len(v): {len(v)}")
    #for hr in  ['1.0', '0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1', '0.09', '0.08', '0.07', '0.06', '0.05', '0.04', '0.03', '0.02', '0.01']:
    ewfd.write('\n')
    for hr in  ['1.0','0.5', '0.1', '0.05']:
        print(f"\t hratio {hr} ")
        ewfd.write(sumdict[k][hr])
