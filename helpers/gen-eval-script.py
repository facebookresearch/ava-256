# FIX: add conda activate in sh command
# ~/0/0/aeparams_500000.pt

#eset = 'TONGUE002-HCAMS'

# CHEEK001-HCAMS  CHEEK001-TRCAMS  FREE_FACE-TRCAMS  JAW005-HCAMS  JAW005-TRCAMS  NEUTRAL_PEAK-HCAMS  NEUTRAL_PEAK-TRCAMS  TONGUE002-HCAMS  TONGUE002-TRCAMS
#['1', 'ajd691', 'r0', '116671', '1', 'FAILED', '6-23:56:34', '2022-08-13 11:05:45', '2022-08-20 11:02:19', 'rsclearn1061', 'https://www.internalfb.com/intern/paste/P523884386/', 'https://www.internalfb.com/intern/paste/P523884404/', 'https://www.internalfb.com/intern/paste/P523884418/', '/checkpoint/avatar/jinkyuk/newconda-camab-exp-aug12/ab-bg-7d-r0-20211108-1309-ajd691/run-1-nodes-1-gpus-16-dl-workers/2022-08-13T11_05_42.090773']
# take csv file -- selected items in jlog and return dict of dict {'sid':{abfactor:checkpointpath}}

NOFMASK_FLAG=True


def getdata(dpath=None):
    lines = open(dpath, 'rt').readlines()
    dinfo = dict()
    for ln in lines:

        print(" LINE {}".format(ln))

        line = ln.rstrip()
        toks = line.split(',')
        print(toks)

        sid = toks[1]
        abf = float(toks[4])
        cpath = f'{toks[13]}/0/0/aeparams_500000.pt'
        print(" SID {}   ABF {}   CPATH {}".format(sid, abf, cpath))

        if sid not in dinfo.keys():
            dinfo[sid] = dict()
        dinfo[sid][abf] = cpath
    return dinfo

if __name__ == "__main__":
    dinfo = getdata("./scripts/AB-aug12.csv")



    for esid, ch in dinfo.items():
        print(" SID {}".format(esid))
        for k, v in ch.items():
            print("\t ab {}  path {}".format(k, v))

        # ch={"1.0":"/checkpoint/avatar/jinkyuk/newconda-camab-exp-aug12/ab-bg-7d-r0-20211108-1309-ajd691/run-1-nodes-1-gpus-16-dl-workers/2022-08-13T11_05_42.090773/0/0/aeparams_500000.pt",
        #     "0.5":"/checkpoint/avatar/jinkyuk/newconda-camab-exp-aug12/ab-bg-7d-r0-20211108-1309-ajd691/run-1-nodes-1-gpus-16-dl-workers/2022-08-13T11_05_51.952522/0/0/aeparams_500000.pt",
        #     "0.1":"/checkpoint/avatar/jinkyuk/newconda-camab-exp-aug12/ab-bg-7d-r0-20211108-1309-ajd691/run-1-nodes-1-gpus-32-dl-workers/2022-08-13T11_06_01.575177/0/0/aeparams_500000.pt",
        #     "0.05":"/checkpoint/avatar/jinkyuk/newconda-camab-exp-aug12/ab-bg-7d-r0-20211108-1309-ajd691/run-1-nodes-1-gpus-64-dl-workers/2022-08-13T11_06_11.368529/0/0/aeparams_500000.pt"}

        if NOFMASK_FLAG == False:
            wfd = open(f'./scripts/eval-camab-aug12-{esid}.sh', 'wt') # generate one sh file for an identity and all test cases
        else:
            wfd = open(f'./scripts-nofmask/eval-camab-aug12-{esid}.sh', 'wt') # generate one sh file for an identity and all test cases


        for eset in ["CHEEK001-HCAMS", "CHEEK001-TRCAMS", "FREE_FACE-TRCAMS", "FREE_FACE-HCAMS", "JAW005-HCAMS", "JAW005-TRCAMS", "NEUTRAL_PEAK-HCAMS",  "NEUTRAL_PEAK-TRCAMS",  "TONGUE002-HCAMS",  "TONGUE002-TRCAMS"]:
            if NOFMASK_FLAG == False:
                lines = open('eval-prim-ref.sh', 'rt').readlines()
            else:
                lines = open('eval-prim-ref-nofmask.sh', 'rt').readlines()

            # 0 -- hdir=/checkpoint/avatar/jinkyuk/EVAL-DLREFACTOR-v0-newingest-ajd
            # 1 -- rsc mkdir -p $hdir
            # 2 -- abf=[abf]
            # 3 -- rsc_launcher launch --projects AIRSTORE_CODEC_AVATAR_20211208_0751_BYJ247_FOV_FULL_ORDER_BY_FRAME_CRYPTO AIRSTORE_CODEC_AVATAR_20211119_0812_DDF659_FOV_FULL_ORDER_BY_FRAME_CRYPTO  AIRSTORE_CODEC_AVATAR_20211111_1300_GJB719_FOV_FULL_ORDER_BY_FRAME_CRYPTO AIRSTORE_CODEC_AVATAR_20211110_0752_XDG960_FOV_FULL_ORDER_BY_FRAME_CRYPTO AIRSTORE_CODEC_AVATAR_20211108_1309_AJD691_FOV_FULL_ORDER_BY_FRAME_CRYPTO AIRSTORE_CODEC_AVATAR_20211102_0837_PAC264_FOV_FULL_ORDER_BY_FRAME_CRYPTO AIRSTORE_CODEC_AVATAR_20211101_0822_ZEV994_FOV_FULL_ORDER_BY_FRAME_CRYPTO AIRSTORE_CODEC_AVATAR_20211012_0758_NYG431_FOV_FULL_ORDER_BY_FRAME_CRYPTO -e 'cd /home/$USER/rsc/working/neurvol2-jason/ && python3 sbatch.py -n 1 -t 1 -w 4 --source-dir /home/$USER/rsc/working/neurvol2-jason/  --config-path /home/$USER/rsc/working/neurvol2-jason/ablation-config/eval-dlrefactor-valid-'$sid'/ --batchsize 6 --ablationcamera --holdoutpath /checkpoint/avatar/jinkyuk/read-only/ablation-test/9id-ablation-plans --holdoutratio '$abf' --displayloss --disableddp --checkpoint-root-dir '$hdir' --disableddp --evalcheckpointpath=[evalcheckpointpath] --program-name ddp-eval.py'
            edatapath = f'/checkpoint/avatar/jinkyuk/read-only/ablation-eval/bz2/{eset}'
            for hratio, path in ch.items():
                for i, l in enumerate(lines):
                    if i == 0:
                        l=l.replace('[eset]', f'{eset}')
                        l=l.replace('[esid]', f'{esid}')
                    if i == 2:
                        l=l.replace('[abf]', f'"{hratio}"')
                    if i == 3:
                        l=l.replace('[evalcheckpointpath]', f"{path}")
                        l=l.replace('[evaldatapath]', f'{edatapath}')
                    print(f" {i} -- {l} ")
                    wfd.write(l)
                wfd.write('\n')
        wfd.close()
