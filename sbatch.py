import copy
import argparse
import datetime
import json
import os
import shlex
import shutil
import subprocess
import sys

from typing import Any, Dict, List, Optional


def get_parser():
    parser = argparse.ArgumentParser("Script for launching UA experiments")

    user = os.environ.get("USER")

    parser.add_argument(
        "--source-dir",
        type=str,
        default=os.getcwd(),
        help="source tree where we assume training code exists"
    )
    parser.add_argument(
        "--checkpoint-root-dir",
        type=str,
        default=f"/checkpoint/avatar/{user}/ua-mvp-runs",
        help="root dir for all checkpoints and other training artifacts"
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="checkpoint dir of the run that needs to be resumed"
    )

    parser.add_argument(
        "--localnode",
        type=str,
        default=None,
        help="checkpoint dir of the run that needs to be resumed"
    )

    parser.add_argument(
        "--requeue-dir",
        type=str,
        default=None,
        help="checkpoint dir of the run that needs to be requeued"
    )
    parser.add_argument(
        "--evalcheckpointpath",
        type=str,
        default=None,
        help="checkpoint dir of the run that needs to be requeued"
    )
    parser.add_argument(
        "--evaldatapath",
        type=str,
	    default=None,
	    help="dir for evaluation data"
    )
    parser.add_argument(
        "-g",
        "--gpus-per-node",
        type=int,
        default=8,
        help="number of GPUs per node"
    )
    parser.add_argument(
        "--ids2use",
        type=int,
        default=-1,
        help="the number of identities in idfile. When it's set to N,  ids[0] .. ids[N-1] are used.  If it's not set, all ids in idfile are used by default."
    )
    parser.add_argument(
        "-t",
        "--ntasks-per-node",
        type=int,
        default=8,
        help="number of tasks per node" # for DDP : set to 8 by default, single GPU training: set to 1
    )
    parser.add_argument(
        "-n",
        "--num-nodes",
        type=int,
        default=1,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "-x",
        "--gpuspertask",
        type=int,
        default=1,
        help="number of GPUs per task"
    )
    parser.add_argument(
        "-c",
        "--cpuspertask",
        type=int,
        default=32,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "-w",
        "--num-airstore-workers",
        type=int,
        default=4,
        help="number of airstore dataloader workers",
    )
    parser.add_argument(
        "--crank",
        type=int,
        default=-1,
        help="chunk rank",
    )
    parser.add_argument(
        "--csize",
        type=int,
        default=-1,
        help="number of chunks",
    )

    parser.add_argument(
        "--rowbegin",
        type=int,
        default=-1,
        help="beginning row id to open",
    )

    parser.add_argument(
        "--rowend",
        type=int,
        default=-1,
        help="ending row id to open",
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="optional relative path for config.py, if default one is not applicable",
    )
    parser.add_argument(
        "--program-name",
        type=str,
        default="ddp-train.py",
        help="python script to run in srun_wrapper.sh",
    )
    parser.add_argument(
        "--holdoutpath",
        type=str,
        default=None,
        help="path for ablation test hold out information",
    )

    parser.add_argument(
        "--finetunefile",
        type=str,
        default=None,
        help="path of finetune configuration file -- finetune cameras and finetune segmentation",
    )

    parser.add_argument(
        "--idfilepath",
        type=str,
        default=None,
        help="path to identity list file",
    )
    parser.add_argument(
        "--disableddp",
        action="store_true",
        default=False,
        help="Flag for controlling whether to disable ddp or not. If disabled, DDP init and DDP primitives are not used at all",
    )
    parser.add_argument(
        "--enabledeterministic",
        action="store_true",
        default=False,
        help="force prefixed seed, by default, False",
    )
    parser.add_argument(
        "--disableshuffle",
        action="store_true",
        default=False,
        help="turn off shuffling in airstore. by defalut, False",
    )

    parser.add_argument(
        "--shard_air",
        action="store_true",
        default=False,
        help="turn on sharding airstore by defalut, False",
    )

    parser.add_argument(
        "--seed_air",
        type=str,
        default=None,
        help="seed for airstore shuffling. enable determinisitic should be set together",
    )

    parser.add_argument(
        "--enablepngparsing",
        action="store_true",
        default=False,
        help="only for scan data experiment -- enable png scanning on samples from AIRSTORE",
    )
    parser.add_argument(
        "--displayloss",
        action="store_true",
        default=False,
        help="whether show loss value every iteration or not",
    )
    parser.add_argument(
        "--displayprofstats",
        action="store_true",
        default=False,
        help="whether log and show iteratiom times every iteration or not. Setting this to true requires periodic cuda synchronization, which will slow down training.",
    )
    parser.add_argument(
        "--prefetch-training-data",
        action="store_true",
        default=False,
        help="Flag for controlling whether to prefetech training data, mainly for debug profiling",
    )
    parser.add_argument(
        "--ablationcamera",
        action="store_true",
        default=False,
        help="Flag for controlling whether to run ablation test or not for camera",
    )
    parser.add_argument(
        "--reservation",
        type=str,
        default=None,
        help="optionally pass a slurm reservation",
    )

    parser.add_argument(
        "--batchsize",
        type=int,
        default=6,
        help="batch size per GPU",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate base at batch size of 8. Will be linearly scaled",
    )

    parser.add_argument(
        "--subsample-size",
        type=int,
        default=2048 // 4,
        help="Size of image after cropping -- or other subsampling method used",
    )

    parser.add_argument(
        "-d",
        "--downsample",
        type=int,
        default=4, #  W=667, H=1024 by default
        help="DL return image size W, H = [2668, 4096] // downsample ",
    )
    parser.add_argument(
        "--holdoutratio",
        type=str,
        default=None,
        #default='1.0',
        help="ablation factor 1.0 = 100 percent cams, 0.1 : 10 percent cams per frame",
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=1_000_000,
        help="max iterations",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="optimizer, defaults to adam",
    )

    parser.add_argument(
        "--tensorboard-logdir",
        type=str,
        default=None,
        help="directory where tensorboard will save its outputs. None means no outputs will be created",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="MGR",
        help="dataset to use",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="cfg node",
    )

    parser.add_argument(
        '--masterport',
        type=str,
        default="43321",
        help='master node network port'
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Interactive debugging on a single GPU",
    )
    parser.add_argument(
        "--disable_id_encoder",
        action='store_true',
        help="disable id_encoder in ae",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=1,
        help="steps to accumulate gradients before updating model weights"
    )
    parser.add_argument(
        "--nids",
        type=int,
        default=1,
        help="number of identities to train on"
    )
    parser.add_argument(
        "--expname",
        type=str,
        default="",
        help="exp name that will be used as prefix in job folder",
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=None,
        help="learning rate for encoder ",
    )
    parser.add_argument(
        "--logallrankstb",
        action='store_true',
        default = None,
        help="Log tensorboard from all ranks, not just rank 0",
    )
    parser.add_argument(
        "--encoder_channel_mult",
        type=int,
        default=1,
        help="Channel multiplier for the encoder"
    )
    return parser


def get_args(input_args: Optional[List[str]] = None):
    """
    input_args (List[str]): strings to parse, defaults to sys.argv
    """
    parser = get_parser()
    args = parser.parse_args(input_args)
    return args

def run_hash(slurmarraytaskid: Optional[int] = None):
    """
    Create a hash for the job. If a slurm array task id is passed, simply concatenate it to a string.
    Otherwise, create current time in ISO format and remove colons, since they cannot be escaped
    in POSIX PATH env vars.
    """
    if slurmarraytaskid is None:
        return datetime.datetime.now().isoformat().replace(":", "_")
    else:
        return f'slurmarray-task-id-{slurmarraytaskid}'

def run_dir(args, run_hash: Optional[str] = None):
    dir = f"run-{args.num_nodes}-nodes-{args.gpus_per_node*args.num_nodes}-gpus-{args.num_airstore_workers}-dl-workers"
    if run_hash is not None:
        # Array jobs don't need a run_hash
        dir = dir + f"/{run_hash}"
    return dir

def slurm_snapshot_code_dir(exp_dir):
    return os.path.join(exp_dir, "slurm_snapshot_code")

def gen_job_config(args):
    config = {}
    config["nodes"] = args.num_nodes
    config["gpus_per_node"] = args.gpus_per_node
    config["ntasks_per_node"] = args.ntasks_per_node
    config["ids2use"] = args.ids2use
    config["batchsize"] = args.batchsize
    config["num_airstore_workers"] = args.num_airstore_workers
    config["num_iterations"] = 0
    config['config_path'] = args.config_path
    config['maxiter'] = args.maxiter
    config['rowbegin'] = args.rowbegin
    config['rowend'] = args.rowend
    config['program_name'] = args.program_name
    config['holdoutpath'] = args.holdoutpath
    config['finetunefile'] = args.finetunefile
    config['idfilepath'] = args.idfilepath
    config['learning_rate'] = args.learning_rate
    config['subsample_size'] = args.subsample_size
    config['downsample'] = args.downsample
    config['holdoutratio'] = args.holdoutratio
    config['crank'] = args.crank
    config['csize'] = args.csize
    config['disableddp'] = args.disableddp
    config['cpuspertask'] = args.cpuspertask
    config['gpuspertask'] = args.gpuspertask
    config['enabledeterministic'] = args.enabledeterministic
    config['disableshuffle'] = args.disableshuffle
    config['shard_air'] = args.shard_air
    config['seed_air'] = args.seed_air
    config['optimizer'] = args.optimizer
    config['tensorboard_logdir'] = args.tensorboard_logdir
    config['slurm_reservation'] = args.reservation
    config['masterport'] = args.masterport
    config['disable_id_encoder'] = args.disable_id_encoder
    config['encoder_lr'] = args.encoder_lr
    config['nids'] = args.nids
    config['logallrankstb'] = args.logallrankstb
    config['encoder_channel_mult'] = args.encoder_channel_mult
    return config

def log_job_config(job_config, slurm_log_dir=None):
    # TODO(julieta) remove dependency on env variable to increase encapsulation
    if slurm_log_dir is None:
        slurm_log_dir = os.environ["RSC_EXP_RUN_BASE_DIR"]
    config_file_name = os.path.join(slurm_log_dir, "job-config.json")
    with open(config_file_name, 'w') as f:
        json.dump(job_config, f)

def read_job_config(config_path):
    config_dict = None
    config_file_name = os.path.join(config_path, "job-config.json")
    if os.path.isfile(config_file_name):
        with open(config_file_name) as config_file:
            config_dict = json.load(config_file)
    return config_dict

def code_snapshot(src_path, dst_path):
    #print(f"{src_path} -- {dst_path}")
    # copy_and_overwrite
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    shutil.copytree(src_path, dst_path)

def setup_job(args):
    if args.requeue_dir:
        exp_run_dir = args.requeue_dir
        code_snapshot_dir = slurm_snapshot_code_dir(exp_run_dir)
        old_job_config = read_job_config(args.requeue_dir)
        batchsize = old_job_config.get("batchsize", -1)
        maxiter = old_job_config.get("maxiter", args.maxiter)
        num_airstore_workers = old_job_config.get("num_airstore_workers", -1)
        config_path = old_job_config.get("config_path", None)
        program_name = old_job_config.get("program_name", args.program_name)
        optimizer = old_job_config.get("optimizer", args.optimizer)
    else:
        exp_hash = run_hash()
        if args.expname:
            exp_hash = f"{args.expname}-{exp_hash}"
        if args.cfg:
            from tune_cfgs.base_cfg import get_cfg_defaults
            cfg = get_cfg_defaults()
            cfg.merge_from_file(args.cfg)

            exp_hash = f"{cfg.expname}-{exp_hash}" # To make the experiment more manageable

        exp_run_leaf_dir = run_dir(args, exp_hash)
        os.makedirs(args.checkpoint_root_dir, exist_ok=True)
        exp_run_dir = os.path.join(args.checkpoint_root_dir, exp_run_leaf_dir)
        os.makedirs(exp_run_dir, exist_ok=True)
        code_snapshot_dir = slurm_snapshot_code_dir(exp_run_dir)
        os.makedirs(code_snapshot_dir, exist_ok=True)
        code_snapshot(args.source_dir, code_snapshot_dir)
        batchsize = args.batchsize
        maxiter = args.maxiter
        num_airstore_workers = args.num_airstore_workers
        config_path = args.config_path
        program_name = args.program_name
        optimizer = args.optimizer

    if args.debug:
        code_snapshot_dir = f"/home/{os.environ['USER']}/rsc/neurvol2-jason"
        os.environ["RSC_DEBUG"] = "true"

    os.environ["RSC_AVATAR_PYUTILS_PATH"] = "/checkpoint/avatar/jinkyuk/pyutils"
    os.environ["RSC_AVATAR_METADATA_PATH"] = "/checkpoint/avatar/jinkyuk/meta-data"
    os.environ["RSC_AVATAR_RSCASSET_PATH"] = "/checkpoint/avatar/jinkyuk/rsc-assets"
    os.environ["RSC_AVATAR_READONLY_PATH"] = "/checkpoint/avatar/jinkyuk/read-only"
    os.environ["RSC_AVATAR_DEBUGDATA_PATH"] = "/checkpoint/avatar/jinkyuk/debug-data"
    os.environ["RSC_AVATAR_EVAL_CONFIG_PATH"] = code_snapshot_dir
    conda_prefix = os.environ.get("CONDA_PREFIX")
    # ld_library_path = f"{conda_prefix}/lib:"
    # ld_library_path += f"{conda_prefix}/lib/python3.8/site-packages:"
    # ld_library_path += "/checkpoint/avatar/ua_setup/vulkan_sdk/1.1.108.0/x86_64/lib:"
    # os.environ["LD_LIBRARY_PATH"] = ld_library_path
    os.environ["RSC_EXP_RUN_BASE_DIR"] = exp_run_dir
    os.environ["RSC_RUN_SLURM_SNAPSHOT_DIR"] = code_snapshot_dir
    os.environ["RSC_UA_MINI_BATCH_SIZE"] = str(batchsize)
    os.environ["RSC_UA_OPTIMIZER"] = optimizer
    os.environ["RSC_UA_NUM_AIRSTORE_WORKERS"] = str(num_airstore_workers)
    os.environ["RSC_CONFIG_DIR"] = code_snapshot_dir
    if config_path:
        os.environ["RSC_CONFIG_DIR"] = os.path.join(code_snapshot_dir, config_path)
    os.environ["RSC_PROGRAM_NAME"] = program_name  # by default, it's set to ddp-train.py

    os.environ["RSC_GPUS_PER_TASK"] = str(args.gpuspertask)

def gen_sbatch_command_and_str(curr_job_config, srun_cmd_str):

    excluded_hosts = os.environ.get("EXCLUDED_HOSTS", None)
    included_hosts = os.environ.get("INCLUDED_HOSTS", None)
    slurm_log_dir = os.environ["RSC_EXP_RUN_BASE_DIR"]
    num_nodes = curr_job_config["nodes"]
    gpus_per_node = curr_job_config["gpus_per_node"]
    ntasks_per_node = curr_job_config["ntasks_per_node"]
    cpuspertask = curr_job_config["cpuspertask"]
    gpuspertask = curr_job_config["gpuspertask"]
    slurm_reservation = curr_job_config["slurm_reservation"]

    print(" GPUSPERTASK : {}".format(gpuspertask))
    print(" GPUSPERTASK TYPE : {}".format(type(gpuspertask)))

    if gpuspertask > 1 and ntasks_per_node > 1:
        assert ntasks_per_node == 8 // gpuspertask

    # always set tasks per node to be same as gpus per node
    # DISABLE THIS
    #if ntasks_per_node != gpus_per_node:
    #    ntasks_per_node = gpus_per_node
    sbatch_cmd = [
        "sbatch",
        #"--gpus",
        #str(gpus_per_node * num_nodes),
        #f"--gres=gpu:{gpuspertask}",
        f"--gres=gpu:{gpus_per_node}",
        #str(gpus_per_node * num_nodes),
        "--nodes",
        str(num_nodes),
        "--ntasks-per-node",
        str(ntasks_per_node),
        #"--gpus-per-task",
        #str(gpuspertask), # 32 for DGX A100 servers
        "--cpus-per-task",
        str(cpuspertask), # 32 for DGX A100 servers
        "--open-mode", # SET OPEN MODE to APPEND (alternative one is "truncate"
        "append",
        "--signal",
        "B:USR1@180",
        "--time",
        "7-00:00:00",
        "--output",
        f"{os.path.join(slurm_log_dir, 'sbatch-%j.out')}",
        "--error",
        f"{os.path.join(slurm_log_dir, 'sbatch-%j.err')}",
        "--exclude=avalearn1088",
    ]
    sbatch_cmd += ["--partition", "learn"]
    sbatch_cmd += ["--mem-per-gpu", "225G"]

    # TODO(julieta) make this optional
    # sbatch_cmd += ["--account=avatar_ai_rsc", "--qos=avatar_ai_rsc_dev"]

    wrapped_cmd = f"{srun_cmd_str}"
    sbatch_cmd += ["--wrap", wrapped_cmd]
    sbatch_cmd_str = " ".join(map(shlex.quote, sbatch_cmd))
    return sbatch_cmd, sbatch_cmd_str

def gen_debug_sbatch_command_and_str(curr_job_config, srun_cmd_str):

    excluded_hosts = os.environ.get("EXCLUDED_HOSTS", None)
    included_hosts = os.environ.get("INCLUDED_HOSTS", None)
    slurm_log_dir = os.environ["RSC_EXP_RUN_BASE_DIR"]
    num_nodes = curr_job_config["nodes"]
    gpus_per_node = curr_job_config["gpus_per_node"]
    ntasks_per_node = curr_job_config["ntasks_per_node"]
    cpuspertask = curr_job_config["cpuspertask"]
    slurm_reservation = curr_job_config["slurm_reservation"]

    # always set tasks per node to be same as gpus per node
    # DISABLE THIS
    #if ntasks_per_node != gpus_per_node:
    #    ntasks_per_node = gpus_per_node
    sbatch_cmd = [
        "srun",
        "--gpus",
        str(gpus_per_node * num_nodes),
        "--nodes",
        str(num_nodes),
        "--ntasks-per-node",
        str(ntasks_per_node),
        "--cpus-per-task",
        str(cpuspertask), # 32 for DGX A100 servers
        "--open-mode", # SET OPEN MODE to APPEND (alternative one is "truncate"
        "append",
        "--time",
        "7-00:00:00",
    ]
    sbatch_cmd += ["--partition", "learn"]
    sbatch_cmd += ["--mem-per-cpu", "8G"]
    if slurm_reservation:
        sbatch_cmd += ["--reservation", slurm_reservation]
    os.environ["CMD_DEBUG"] = srun_cmd_str

    # Don't really execute the command but put it in the env so that it can be executed multiple times while debugging.
    sbatch_cmd += ['--pty', 'bash', '-l'] # run bash instead for debugging with repeative runs
    sbatch_cmd_str = " ".join(map(shlex.quote, sbatch_cmd))
    return sbatch_cmd, sbatch_cmd_str

def run_batch(env, sbatch_cmd_str, sbatch_cmd):
    print(f"running command: {sbatch_cmd_str}\n")
    with subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE, env=env) as train_proc:
        stdout = train_proc.stdout.read().decode("utf-8")
        try:
            job_id = int(stdout.rstrip().split()[-1])
            print(f"Launched job {job_id}")
        except IndexError:
            job_id = None
    return job_id, stdout

def can_resume_job(old_job_config, curr_job_config):
    retval = False
    if old_job_config:
        old_nodes = old_job_config.get("nodes", -1)
        curr_nodes = curr_job_config.get("nodes", -2)
        old_gpus_per_node = old_job_config.get("gpus_per_node", -1)
        curr_gpus_per_node = curr_job_config.get("gpus_per_node", -2)
        old_ntasks_per_node = old_job_config.get("ntasks_per_node", -1)
        curr_ntasks_per_node = curr_job_config.get("ntasks_per_node", -2)
        old_batchsize = old_job_config.get("batchsize", -1)
        curr_batchsize = curr_job_config.get("batchsize", -2)
        old_num_airstore_workers = old_job_config.get("num_airstore_workers", -1)
        curr_num_airstore_workers = curr_job_config.get("num_airstore_workers", -2)
        old_config_path = old_job_config.get("config_path", None)
        curr_config_path = curr_job_config.get("config_path", None)
        if old_nodes == curr_nodes and \
           old_gpus_per_node == curr_gpus_per_node and \
           old_ntasks_per_node == curr_ntasks_per_node and \
           old_batchsize == curr_batchsize and \
           old_num_airstore_workers == curr_num_airstore_workers and \
           old_config_path == curr_config_path:
           retval = True
    return retval


def append_args_to_cmdline(
    job_config: Dict[str, Any],
    args: argparse.Namespace,
    cmd_to_append_to: str,
) -> str:

    holdoutpath = job_config.get('holdoutpath', args.holdoutpath)
    finetunefile = job_config.get('finetunefile', args.finetunefile)
    idfilepath = job_config.get('idfilepath', args.idfilepath)
    holdoutratio = job_config.get('holdoutratio', args.holdoutratio)

    crank = job_config.get('crank', args.crank)
    csize = job_config.get('csize', args.csize)
    ids2use = job_config.get('ids2use', args.ids2use)
    disableddp = job_config.get('disableddp', args.disableddp)
    enabledeterministic = job_config.get('enabledeterministic', args.enabledeterministic)

    disableshuffle = job_config.get('disableshuffle', args.disableshuffle)
    shard_air = job_config.get('shard_air', args.shard_air)
    seed_air = job_config.get('seed_air', args.seed_air)

    # wrapper_script_str = f"{os.path.join(code_snapshot_dir, 'srun-wrapper.sh')}"

    srun_cmd_str = f"srun {cmd_to_append_to} " if not args.debug else str(cmd_to_append_to)
    if args.prefetch_training_data:
        srun_cmd_str += " --debugprefetch" # $1 in srun wrapper
    elif args.ablationcamera:
        srun_cmd_str += f" --ablationcamera --holdoutpath {holdoutpath} --holdoutratio {holdoutratio} " # $1, $2 $3

    if disableddp:
        srun_cmd_str += f" --disableddp"

    if enabledeterministic:
        srun_cmd_str += f" --enabledeterministic"

    if disableshuffle:
        srun_cmd_str += f" --disableshuffle"

    if shard_air:
        srun_cmd_str += f" --shard_air"

    if seed_air:
        srun_cmd_str += f" --seed_air {seed_air}"

    if crank != -1:
        srun_cmd_str += f" --crank {crank}"

    if csize != -1:
        srun_cmd_str += f" --csize {csize}"

    if ids2use != -1:
        srun_cmd_str += f" --ids2use {ids2use}"

    if args.displayloss:
        srun_cmd_str += f" --displayloss"

    if args.displayprofstats:
        srun_cmd_str += f" --displayprofstats"

    if args.evalcheckpointpath:
        srun_cmd_str += f" --evalcheckpointpath {args.evalcheckpointpath}"

    if args.batchsize:
        srun_cmd_str += f" --batchsize {args.batchsize}"

    if args.subsample_size:
        srun_cmd_str += f" --subsample-size {args.subsample_size}"

    if args.learning_rate:
        srun_cmd_str += f" --learning-rate {args.learning_rate}"

    if args.evaldatapath:
        srun_cmd_str += f" --evaldatapath {args.evaldatapath}"

    if idfilepath:
        srun_cmd_str += f" --idfilepath {idfilepath}"

    if finetunefile:
        srun_cmd_str += f" --finetunefile {finetunefile}"

    if args.rowbegin != -1:
        srun_cmd_str += f" --rowbegin {args.rowbegin}"

    if args.rowend != -1:
        srun_cmd_str += f" --rowend {args.rowend}"

    if args.tensorboard_logdir is not None:
        srun_cmd_str += f" --tensorboard-logdir {args.tensorboard_logdir}"

    if args.dataset is not None:
        srun_cmd_str += f" --dataset {args.dataset}"

    if args.cfg is not None:
        srun_cmd_str += f" --cfg {args.cfg}"

    srun_cmd_str += f" --gradient_accumulation {args.gradient_accumulation}"

    if args.disable_id_encoder:
        srun_cmd_str += f" --disable_id_encoder"

    if args.downsample:
        srun_cmd_str += f" --downsample {args.downsample}"

    if args.encoder_lr is not None:
        srun_cmd_str += f" --encoder_lr {args.encoder_lr}"

    if args.nids is not None:
        srun_cmd_str += f" --nids {args.nids}"

    if args.logallrankstb:
        srun_cmd_str += f" --logallrankstb"

    srun_cmd_str += f" --masterport {args.masterport}"

    return srun_cmd_str


def main(scheduler_args: Optional[List[str]] = None):
    args = get_args(scheduler_args)
    print(args)

    setup_job(args)

    log_jconfig = False

    if args.requeue_dir:
        print("all args will be ignored, we will use all parameters from requeue dir")
        curr_job_config = read_job_config(args.requeue_dir)
        os.environ["RSC_JOB_REQUEUE_DIR"] = args.requeue_dir
    else:
        curr_job_config = gen_job_config(args)
        if args.resume_dir:
            old_job_config = None
            can_resume = False
            old_job_config = read_job_config(args.resume_dir)
            if old_job_config:
                can_resume = can_resume_job(old_job_config, curr_job_config)
            if can_resume:
                os.environ["RSC_JOB_RESUME_DIR"] = args.resume_dir
                # reset iterations
                curr_job_config["num_iterations"] = old_job_config["num_iterations"]
            else:
                print(f"cannot resume job - mismatched configs, config of job to resume {old_job_config}, config of curr job {curr_job_config}\n")
                return
        log_jconfig = True

    code_snapshot_dir = os.environ["RSC_RUN_SLURM_SNAPSHOT_DIR"]

    # generate srun command with passed params
    cmd_to_append_to = f"{os.path.join(code_snapshot_dir, 'srun-wrapper.sh')}"
    srun_cmd_str = append_args_to_cmdline(
        curr_job_config,
        args,
        cmd_to_append_to,
    )

    # generate sbatch command
    sbatch_cmd, sbatch_cmd_str = gen_sbatch_command_and_str(curr_job_config, srun_cmd_str) if not args.debug else gen_debug_sbatch_command_and_str(curr_job_config, srun_cmd_str)

    if curr_job_config.get("sbatch_cmd_str", None) is None:
        curr_job_config["sbatch_cmd_str"] = sbatch_cmd_str

    if log_jconfig:
        log_job_config(curr_job_config)

    print(f"srun cmd : {srun_cmd_str}")
    print(f"sbatch cmd: {sbatch_cmd}")
    if args.debug:
        os.system(sbatch_cmd_str)
        return

    if args.localnode == None:
        run_batch(env=os.environ.copy(), sbatch_cmd_str=sbatch_cmd_str, sbatch_cmd=sbatch_cmd)
    else:
        # pass environment variable to remotely run process
        envars=['RSC_AVATAR_PYUTILS_PATH', 'RSC_AVATAR_METADATA_PATH', 'RSC_AVATAR_RSCASSET_PATH', 'RSC_AVATAR_READONLY_PATH', 'RSC_AVATAR_DEBUGDATA_PATH', 'RSC_AVATAR_EVAL_CONFIG_PATH', 'CONDA_PREFIX', 'LD_LIBRARY_PATH', 'RSC_EXP_RUN_BASE_DIR', 'RSC_RUN_SLURM_SNAPSHOT_DIR', 'RSC_UA_MINI_BATCH_SIZE', 'RSC_UA_NUM_AIRSTORE_WORKERS', 'RSC_CONFIG_DIR', 'RSC_CONFIG_DIR', 'RSC_PROGRAM_NAME', 'RSC_JOB_RESUME_DIR', 'RSC_JOB_UUID']
        estring=""
        for var in envars:
            estring += "&& "
            estring += f"export {var}=${var} "

        #replace srun with remote ssh run
        sruncmd = copy.deepcopy(srun_cmd_str)
        tcmd = 'bash'+sruncmd[len('srun'):]
        lcmd = tcmd.replace('srun-wrapper.sh', 'local-wrapper.sh')
        print(" LCMD : {}".format(lcmd))
        print(" local node : {}".format(args.localnode))

        #cmd = f'ssh {args.localnode} "module load anaconda3/2021.05  && source /checkpoint/avatar/conda-envs/ua-env-20220804-airstore/bin/activate {estring} && export RSC_JOB_UUID=$RSC_JOB_UUID && echo $RSC_JOB_UUID && cd /home/jinkyuk/rsc/working/neurvol2-jason  && {lcmd}"'
        cmd = f'ssh {args.localnode} "module load anaconda3/2021.05  && source /checkpoint/avatar/conda-envs/ua-env-20220804-airstore/bin/activate {estring} && export RSC_JOB_UUID=$RSC_JOB_UUID && echo $RSC_JOB_UUID && cd /home/jinkyuk/rsc/clean-master/neurvol2-jason  && {lcmd}"'

        print("@@@@@@@@@@@@@@@@@@@@@@@@ {}".format(cmd))
        os.system(cmd)

if __name__ == "__main__":
    main(sys.argv[1:])
