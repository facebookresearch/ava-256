# SBATCH CMD

1. sbatch -N [NODE CNT] --no-requeue run.sh [IO worker per GPU] [config file path]
2. Two config files for with/without profiling
- with pytorching profiling
sbatch -N 2 --no-requeue run.sh 4 config-profile.py
- without pytorch profiling
sbatch -N 2 --no-requeue run.sh 4 config.py
3. Disable/Enable CPU prefetching in emacs srun-wrapper.sh : remove debugprefetch option
"CUDA_VISIBLE_DEVICES=${SLURM_LOCALID} python ddp-train.py ${RUN} -numworker $1 --debugprefetch"


Example launch command for ablation exp's 3 stages: traing, finetuning, and test.

```
# Stage-1: training
CRYPTO=AIRSTORE_CODEC_AVATAR_20211123_1247_GKH125_FOV_FULL_ORDER_BY_FRAME_DEC_CRYPTO
workers=30
cpus=32
abf=1.0
sid=20211123-1247-gkh125
ver='0' # repetition number
hdir=/checkpoint/avatar/$USER/ablation/training-${sid}-${ver}
rsc mkdir -p $hdir

rsc_launcher launch --projects ${CRYPTO} -e 'module load anaconda3/2021.05  && source /checkpoint/avatar/conda-envs/ua-env-20220804-airstore/bin/activate  && cd /home/$USER/rsc/test-master/neurvol2-jason/ && python3 sbatch.py -g 1 -n 1 -t 1 -c '$cpus' -w '$workers' --source-dir /home/$USER/rsc/test-master/neurvol2-jason/  --config-path /home/$USER/rsc/test-master/neurvol2-jason/ --batchsize 6 --ablationcamera --holdoutpath /checkpoint/avatar/jinkyuk/read-only/ablation-plans/per-version/'${ver}' --holdoutratio '$abf' --enabledeterministic --displayloss --disableddp --checkpoint-root-dir '$hdir' --disableddp --shard_air --idfilepath /home/$USER/rsc/test-master/neurvol2-jason/ablation-config/cam-ab2/'$sid' --seed_air '${ver}${CRYPTO}${abf}
```

```
# Stage-2: finetuning
sid=20211123-1247-gkh125
ver='0' # repetition number
CRYPTO=AIRSTORE_CODEC_AVATAR_20211123_1247_GKH125_FOV_FULL_ORDER_BY_FRAME_DEC_CRYPTO
finetunedir=/checkpoint/avatar/$USER/ablation/finetune-${sid}-${ver}
rsc mkdir -p $finetunedir
evalcheckpoint=[You need to put a path to training checkpoint you want to use for finetuning]
rowbegin=[You need to  find hive row range for a test segment using ./helpers/gen-begin-end-for-segments.py]
rowend=[You need to  find hive row range for a test segment using ./helpers/gen-begin-end-for-segments.py]

rsc_launcher launch --projects ${CRYPTO} -e 'module load anaconda3/2021.05 && source /checkpoint/avatar/conda-envs/ua-env-20220804-airstore/bin/activate && cd /home/$USER/rsc/test-master/neurvol2-jason/ && python3 sbatch.py -n 1 -g 1 -t 1 -c 32 -w 30 --source-dir /home/$USER/rsc/test-master/neurvol2-jason/ --checkpoint-root-dir '${finetunedir}' --idfilepath /home/$USER/rsc/test-master/neurvol2-jason/ablation-config/cam-ab2/'$sid' --displayloss --displayprofstats --enabledeterministic --seed_air test-0 --shard_air --finetunefile /checkpoint/avatar/jinkyuk/read-only/finetune-plan/finetune.npy --evalcheckpointpath '${evalcheckpoint}' --rowbegin '${rowbegin}' --rowend '${rowend}$' --program-name ddp-finetune-rsc.py --disableddp'
```

```
# Stage-3: evaluation
sid=20211123-1247-gkh125
ver='0' # repetition number
CRYPTO=AIRSTORE_CODEC_AVATAR_20211123_1247_GKH125_FOV_FULL_ORDER_BY_FRAME_DEC_CRYPTO
evaldir=/checkpoint/avatar/$USER/ablation/evaluate-${sid}-${ver}
rsc mkdir -p ${evaldir}
rowbegin=[You need to  find hive row range for a test segment using ./helpers/gen-begin-end-for-segments.py]
rowend=[You need to  find hive row range for a test segment using ./helpers/gen-begin-end-for-segments.py]
evalcheckpoint=[You need to put a path to training checkpoint you want to use for finetuning]

rsc_launcher launch --projects ${CRYPTO} -e 'module load anaconda3/2021.05 && source /checkpoint/avatar/conda-envs/ua-env-20220804-airstore/bin/activate && cd /home/$USER/rsc/test-master/neurvol2-jason/ && python3 sbatch.py -n 1 -g 1 -t 1 -w 30 --source-dir /home/$USER/rsc/test-master/neurvol2-jason/ --checkpoint-root-dir '${evaldir}' --idfilepath /home/$USER/rsc/commit-master/neurvol2-jason/ablation-config/cam-ab2/'$sid' --displayloss --displayprofstats --enabledeterministic --seed_air test-1 --shard_air --finetunefile /checkpoint/avatar/jinkyuk/read-only/finetune-plan/evaluate.npy --evalcheckpointpath '${evalcheckpoint}' --rowbegin '${rowbegin}'  --rowend '${rowend}' --program-name ddp-eval-rsc.py --disableddp --batchsize 1 --maxiter=1000 --disableshuffle'
```

```
# EXAMPLE OF FINETUNE and EVALUATION
# FINETUNE EXAMPLE :
# gks:  min row id 2122913, max row id 2275969 for EXP_free_face
rsc_launcher launch --projects AIRSTORE_CODEC_AVATAR_20211123_1247_GKH125_FOV_FULL_ORDER_BY_FRAME_FLASHARRAY_CRYPTO -e 'module load anaconda3/2021.05 && source /checkpoint/avatar/conda-envs/ua-env-20220804-airstore/bin/activate && cd /home/$USER/rsc/dev-master/neurvol2-jason/ && python3 sbatch.py -n 1 -g 1 -t 1 -w 4 --source-dir /home/$USER/rsc/dev-master/neurvol2-jason/ --checkpoint-root-dir /checkpoint/avatar/$USER/dev-master/finetune-gkh --idfilepath /home/$USER/rsc/commit-master/neurvol2-jason/ablation-config/cam-ab2/20211123-1247-gkh125 --displayloss --displayprofstats --enabledeterministic --seed_air test-1 --shard_air --finetunefile /checkpoint/avatar/jinkyuk/read-only/finetune-plan/finetune.npy --evalcheckpointpath /checkpoint/avatar/jinkyuk/nov18-newabplan-copy/test-0-20211123-1247-gkh125/run-1-nodes-1-gpus-16-dl-workers/2022-11-18T20_57_41.335508/0/0/aeparams_1424000.pt --rowbegin 2122913 --rowend 2275969 --program-name ddp-finetune-rsc.py --disableddp'
```

```
# EVALUATION EXAMPLE :
evalcheckpoint=/checkpoint/avatar/jinkyuk/dev-master/finetune-gkh/run-1-nodes-1-gpus-60-dl-workers/2022-12-09T18_04_36.476891/0/0/aeparams_050000.pt

rsc_launcher launch --projects AIRSTORE_CODEC_AVATAR_20211123_1247_GKH125_FOV_FULL_ORDER_BY_FRAME_FLASHARRAY_CRYPTO -e 'module load anaconda3/2021.05 && source /checkpoint/avatar/conda-envs/ua-env-20220804-airstore/bin/activate && cd /home/$USER/rsc/dev-master/neurvol2-jason/ && python3 sbatch.py -n 1 -g 1 -t 1 -w 30 --source-dir /home/$USER/rsc/dev-master/neurvol2-jason/ --checkpoint-root-dir /checkpoint/avatar/$USER/dev-master/DEBUG-nofinetune-evaluate-gkh-EXP_cheek001e-50k-finetune-nosubsample --idfilepath /home/$USER/rsc/commit-master/neurvol2-jason/ablation-config/cam-ab2/20211123-1247-gkh125 --displayloss --displayprofstats --enabledeterministic --seed_air test-1 --shard_air --finetunefile /checkpoint/avatar/jinkyuk/read-only/finetune-plan/evaluate.npy --evalcheckpointpath '$evalcheckpoint' --rowbegin 994300  --rowend 1005699 --program-name ddp-eval-rsc.py --disableddp \
--batchsize 1 --maxiter=1000 --disableshuffle'
```


# MULTIPLE IDENTITY SUPPORT:

## Stage-1: training with Multiple identities:

```
#Example Command:
rsc_launcher launch --projects AIRSTORE_CODEC_AVATAR_20211123_1247_GKH125_FOV_FULL_ORDER_BY_FRAME_DEC_CRYPTO AIRSTORE_CODEC_AVATAR_20211119_0812_DDF659_FOV_FULL_ORDER_BY_FRAME_DEC_CRYPTO  -e 'module load anaconda3/2021.05  && source /checkpoint/avatar/conda-envs/ua-env-20220804-airstore/bin/activate  && cd /home/$USER/rsc/test-master//neurvol2-jason/ && python3 sbatch.py -g 1 -n 1 -t 1 -c '$cpus' -w '$workers' --source-dir /home/$USER/rsc/test-master/neurvol2-jason/  --config-path /home/$USER/rsc/test-master/neurvol2-jason/ --batchsize 6 --ablationcamera --holdoutpath /checkpoint/avatar/jinkyuk/read-only/ablation-plans/per-version/'${ver}' --holdoutratio '$abf' --enabledeterministic --displayloss --disableddp --checkpoint-root-dir '$hdir' --disableddp --shard_air --idfilepath /home/$USER/rsc/test-master/neurvol2-jason/ablation-config/cam-ab2/2ids.txt'' --seed_air '${ver}'2ids'${abf}

#2ids.txt:
#codec_avatar_20211119_0812_ddf659_fov_full_order_by_frame_dec
#codec_avatar_20211123_1247_gkh125_fov_full_order_by_frame_dec
```


# SLURM ARRAY COMMAND SUPPORT

Just run `python run-array.py`, the tool will by default launch two jobs in an array for you. Feel free to customize it to your needs (eg sweeping).


# RUN without airstore's sharding  IN AVA RSC

SCENV=ava mkdir /checkpoint/avatar/jinkyuk/mvp-runs-nosharding-master-32node-mb32-w4-lre-6-nodeterministic
SCENV=ava rsc_launcher launch --projects codec_avatar_20210504_0803_edl430_fov_full_order_by_frame_april -e 'source /checkpoint/avatar/conda-envs/ua-env-20220804-airstore/bin/activate  && cd /home/$USER/rsc/neurvol2-jason/ && python3 sbatch.py -g 8 -n 32 -t 8 -w 4 --source-dir /home/$USER/rsc/neurvol2-jason/ --ablationcamera --holdoutpath /checkpoint/avatar/jinkyuk/read-only/ablation-plans/per-version/0 --holdoutratio 1.0 --optimizer adam --displayloss --displayprofstats --checkpoint-root-dir /checkpoint/avatar/jinkyuk/mvp-runs-nosharding-master-32node-mb8-w4-lre-6-nodeterministic  --idfilepath /checkpoint/avatar/jinkyuk/idfiles/EDL430-20210504-0803 --reservation meta_testing --batchsize 8'



# DESCRIPTIONS FOR FEW CRITICAL ARGUMENTS
- n: # of nodes
- t: task per node
- g: gpus per node (t EQ g)
- c: cpus per task
- w: number of data loader workers ( w-1 <= c)
- batchsize : batch size per GPU
- enabledeterministic: enforce predefined seed for airstore shuffling
- shard_air: enable airstore sharding
- seed_air: seed for airstore shuffling

For enabling airstore sharding properly, the commandline should INCLUDE "--enabledeterministic  --shard_air --seed_air [randomstring]"
For disable airstore sharding properly, the commandline should REMOVE "--enabledeterministic  --shard_air --seed_air [randomstring]"


# DROP ablation related configurations and skip using ablation meta dta
# RUN a training job with 64 newly ingested png data sets

CRYPT='AIRSTORE_CODEC_AVATAR_20210223_1023_AVW368_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210224_1237_RTO934_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210405_1000_ROW429_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210504_0803_EDL430_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210510_1347_PCO068_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210511_0825_ROW429_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210511_1400_YBN667_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210512_1331_GUI516_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210519_0856_VKE790_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210520_1343_BYJ247_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210525_0843_UNW124_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210526_1356_XLC201_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210527_0833_BEW088_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210527_1341_FAN949_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210602_1332_EZI184_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210603_0835_NRF398_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210604_0814_RTO934_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210615_0845_HJT550_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210616_0923_EPT333_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210617_0851_YQM749_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210618_0851_LFU807_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210618_1330_TID371_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210621_0846_RWE096_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210621_1353_BNP056_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210623_0846_QKY091_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210624_1334_LVA891_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210628_1321_HHR586_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210629_0936_OJD275_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210629_1304_TOV784_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210630_0850_CSM352_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210630_1323_PZW158_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210701_0830_XIV717_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210716_0822_ZJH465_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210716_1406_NHH376_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210719_0822_WSO766_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210719_1411_OEX299_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210720_0832_UTW047_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210726_1346_BMS707_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210727_1338_ZGE795_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210728_1314_PMS851_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210806_0820_LLC960_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210809_1318_MFS088_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210811_0809_BNP056_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210812_0808_CZO827_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210812_1337_UZL880_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210816_1328_BLY655_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210817_0900_NRE683_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210817_1332_GYQ638_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210818_1332_CDR970_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210819_0903_DOT682_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210820_0841_XPU211_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210820_1341_VRH652_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210823_1324_CTR406_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210824_1337_LVA891_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210825_1323_CEK348_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210826_1341_CQL094_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210827_1410_FTO337_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210830_0821_JED279_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210830_1345_QMA167_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210831_0815_XJT672_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210901_0833_LAS440_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210902_0817_IFG774_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210902_1408_DRK347_UCA1_PNG_NO_USER_DATA_CRYPTO AIRSTORE_CODEC_AVATAR_20210917_0825_SES960_UCA1_PNG_NO_USER_DATA_CRYPTO '

DS=/checkpoint/avatar/jinkyuk/idfiles/64ids-m0-png.txt
SCENV=ava mkdir /checkpoint/avatar/jinkyuk/m0-opent-retry-opent/64ids
SCENV=ava rsc_launcher launch --projects $CRYPT -e 'source /checkpoint/avatar/conda-envs/ua-env-20220804-airstore/bin/activate  && cd /home/$USER/rsc/neurvol2-jason/ && python3 sbatch.py -g 8 -n 1 -t 8 -w 4 --source-dir /home/$USER/rsc/neurvol2-jason/ --batchsize 8 --optimizer adam --displayloss --displayprofstats --checkpoint-root-dir /checkpoint/avatar/jinkyuk/m0-opent-retry-opent/64ids  --idfilepath /checkpoint/avatar/jinkyuk/idfiles/64ids-m0-png.txt  --reservation meta_testing --maxiter 1000000'
SCENV=ava rsc_launcher launch --projects $CRYPT -e 'source /checkpoint/avatar/conda-envs/ua-env-20220804-airstore/bin/activate  && cd /home/$USER/rsc/neurvol2-jason/ && python3 sbatch.py -g 8 -n 4 -t 8 -w 4 --source-dir /home/$USER/rsc/neurvol2-jason/ --batchsize 8 --optimizer adam --displayloss --displayprofstats --checkpoint-root-dir /checkpoint/avatar/jinkyuk/m0-opent-retry-opent/64ids  --idfilepath /checkpoint/avatar/jinkyuk/idfiles/64ids-m0-png.txt  --reservation meta_testing --maxiter 1000000'


## Debugging
```bash
# Modifiy this path if the key is at elsewhere
SCENV=ava rsc_launcher launch --projects AIRSTORE_AVATAR_RSC_DATA_PIPELINE_CRYPTO

export RSC_JOB_UUID=...

source /uca/conda-envs/activate-latest  \
&& cd /home/$USER/rsc/neurvol2-jason/ && python3 sbatch.py -g 1 -n 1 -t 1 -w 1 \
--source-dir /home/$USER/rsc/neurvol2-jason/ \
--batchsize 4 \
--subsample-size 512 \
--learning-rate 0.001 \
--optimizer adam \
--displayloss \
--checkpoint-root-dir /uca_transient_a/$USER/ \
--tensorboard-logdir /uca_transient_a/$USER/tensorboard/ \
--maxiter 500000 \
--disableddp \
--dataset=mugsy \
--debug --disable_id_encoder

#Run command with  after get into a new machine
$CMD_DEBUG
```

## Training

```bash
SCENV=ava rsc_launcher launch \
--projects AIRSTORE_AVATAR_RSC_DATA_PIPELINE_CRYPTO -e \
'source /uca/conda-envs/activate-latest  \
&& cd /home/$USER/rsc/neurvol2-jason/ && python3 sbatch.py -g 8 -n 1 -t 8 -w 4 \
--source-dir /home/$USER/rsc/neurvol2-jason/ \
--batchsize 4 \
--subsample-size 512 \
--learning-rate 0.001 \
--optimizer adam \
--displayloss \
--checkpoint-root-dir /uca_transient_a/$USER/ \
--tensorboard-logdir /uca_transient_a/$USER/tensorboard/ \
--maxiter 100000 \
--dataset=mugsy \
--disable_id_encoder'
```
