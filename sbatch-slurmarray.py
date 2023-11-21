# TODO: move job-config to slurm task directory and support MANUAL requeue and resume feature

import copy
import argparse
import datetime
import json
import os
import shlex
import shutil
import subprocess
import sys

from typing import List, Optional

from sbatch import (
    append_args_to_cmdline,
    gen_job_config,
    log_job_config,
    run_hash,
    run_batch,
    run_dir,
    slurm_snapshot_code_dir,
    code_snapshot,
    get_parser as get_parser_sbatch
)


def get_parser():
    parser = get_parser_sbatch()
    # NOTE(julieta) required arguments are more ergonomic if they are positional. Consider refactoring this.
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument(
        "--slurmarraycmdfilepath",
        type=str,
        default=None,
        help="File path where commands will be saved to -- this should be in RSC, not a dev server",
        required=True,
    )
    # TODO(julieta) Infer this from the number of commands that we are launching
    parser.add_argument(
        "--jobcount4slurmarray",
        type=int,
        default=0,
        help="number of slurm jobs for slurm array -- slurm array task id goes from 0 to `jobcount4slurmarray-1`",
        required=True,
    )
    return parser


def get_args(input_args: Optional[List[str]] = None):
    """
    input_args (List[str]): strings to parse, defaults to sys.argv
    """
    parser = get_parser()
    args = parser.parse_args(input_args)
    return args


def gen_job_config(args):
    # TODO(julieta) see if we can share this with sbatch too
    config = {}
    config["nodes"] = args.num_nodes
    config["gpus_per_node"] = args.gpus_per_node
    config["ntasks_per_node"] = args.ntasks_per_node
    config["ids2use"] = args.ids2use
    config["jobcount4slurmarray"] = args.jobcount4slurmarray
    config["batchsize"] = args.batchsize
    config["num_airstore_workers"] = args.num_airstore_workers
    config["num_iterations"] = 0
    config['config_path'] = args.config_path
    config['slurmarraycmdfilepath'] = args.slurmarraycmdfilepath
    config['maxiter'] = args.maxiter
    config['rowbegin'] = args.rowbegin
    config['rowend'] = args.rowend
    config['program_name'] = args.program_name
    config['holdoutpath'] = args.holdoutpath
    config['finetunefile'] = args.finetunefile
    config['idfilepath'] = args.idfilepath
    config['holdoutratio'] = args.holdoutratio
    config['crank'] = args.crank
    config['csize'] = args.csize
    config['disableddp'] = args.disableddp
    config['cpuspertask'] = args.cpuspertask
    config['enabledeterministic'] = args.enabledeterministic
    config['disableshuffle'] = args.disableshuffle
    config['shard_air'] = args.shard_air
    config['seed_air'] = args.seed_air
    config['optimizer'] = args.optimizer
    config['slurm_reservation'] = args.reservation

    return config


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
        exp_run_leaf_dir = run_dir(args)

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

    os.environ["RSC_AVATAR_METADATA_PATH"] = "/checkpoint/avatar/jinkyuk/meta-data"
    os.environ["RSC_AVATAR_RSCASSET_PATH"] = "/checkpoint/avatar/jinkyuk/rsc-assets"
    os.environ["RSC_AVATAR_READONLY_PATH"] = "/checkpoint/avatar/jinkyuk/read-only"
    os.environ["RSC_AVATAR_DEBUGDATA_PATH"] = "/checkpoint/avatar/jinkyuk/debug-data"
    os.environ["RSC_AVATAR_EVAL_CONFIG_PATH"] = code_snapshot_dir
    conda_prefix = os.environ.get("CONDA_PREFIX")
    ld_library_path = f"{conda_prefix}/lib:"
    ld_library_path += f"{conda_prefix}/lib/python3.8/site-packages:"
    ld_library_path += "/checkpoint/avatar/ua_setup/vulkan_sdk/1.1.108.0/x86_64/lib"
    os.environ["LD_LIBRARY_PATH"] = ld_library_path
    os.environ["RSC_EXP_RUN_BASE_DIR"] = exp_run_dir
    os.environ["RSC_RUN_SLURM_SNAPSHOT_DIR"] = code_snapshot_dir
    os.environ["RSC_UA_MINI_BATCH_SIZE"] = str(batchsize)
    os.environ["RSC_UA_OPTIMIZER"] = optimizer
    os.environ["RSC_UA_NUM_AIRSTORE_WORKERS"] = str(num_airstore_workers)
    os.environ["RSC_CONFIG_DIR"] = code_snapshot_dir
    if config_path:
        os.environ["RSC_CONFIG_DIR"] = os.path.join(code_snapshot_dir, config_path)
    os.environ["RSC_PROGRAM_NAME"] = program_name  # by default, it's set to ddp-train.py
    if args.slurmarraycmdfilepath != None:
        print(f"@@@@@@ SLURMARRAYCMDFILEPATH : {args.slurmarraycmdfilepath}")
        os.environ["SLURMARRAYCMDFILEPATH"] = args.slurmarraycmdfilepath


def gen_sbatch_command_and_str(curr_job_config, srun_cmd_str):
    # TODO(julieta) We can share most of this with sbatch

    excluded_hosts = os.environ.get("EXCLUDED_HOSTS", None)
    included_hosts = os.environ.get("INCLUDED_HOSTS", None)
    slurm_log_dir = os.environ["RSC_EXP_RUN_BASE_DIR"]
    num_nodes = curr_job_config["nodes"]
    gpus_per_node = curr_job_config["gpus_per_node"]
    ntasks_per_node = curr_job_config["ntasks_per_node"]
    cpuspertask = curr_job_config["cpuspertask"]
    slurm_reservation = curr_job_config["slurm_reservation"]

    jobcount4slurmarray=curr_job_config["jobcount4slurmarray"]
    assert jobcount4slurmarray > 0
    print(f"gen sbatch-slurmarray command and str : JOBCOUNT4SLURM array {jobcount4slurmarray}")

    # always set tasks per node to be same as gpus per node
    # DISABLE THIS
    #if ntasks_per_node != gpus_per_node:
    #    ntasks_per_node = gpus_per_node
    sbatch_cmd = [
        "sbatch",
        # "--reservation=T128512059.1",
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
        "--signal",
        "B:USR1@180",
        # NOTE(julieta) this is asking for a day and then the job will die. Make this customizable
        "--time",
        "7-00:00:00",
        "--output",
        # f"{os.path.join(slurm_log_dir, 'sbatch-%j.out')}",
        f"{os.path.join(slurm_log_dir, 'sbatch-%A-%a.out')}", # %A: slurm array job id, %a: slurm array task id
        "--error",
        #f"{os.path.join(slurm_log_dir, 'sbatch-%j.err')}",
        f"{os.path.join(slurm_log_dir, 'sbatch-%A-%a.err')}", # %A: slurm array job id, %a: slurm array task id
        "--array",
        f"0-{jobcount4slurmarray-1}"
        #"--no-requeue",
    ]
    sbatch_cmd += ["--partition", "learn"]
    sbatch_cmd += ["--mem-per-cpu", "8G"]
    if slurm_reservation:
        sbatch_cmd += ["--reservation", slurm_reservation]
    wrapped_cmd = f"{srun_cmd_str}"
    sbatch_cmd += ["--wrap", wrapped_cmd]
    sbatch_cmd_str = " ".join(map(shlex.quote, sbatch_cmd))
    return sbatch_cmd, sbatch_cmd_str


def main(scheduler_args: Optional[List[str]] = None):
    args = get_args(scheduler_args)
    print(args)

    setup_job(args)

    # TODO(julieta): support MANUAL requeue and resume feature for slurm array task, or not -- arrays might not be a good fit for this
    assert args.requeue_dir == None
    assert args.resume_dir == None

    curr_job_config = gen_job_config(args)

    jobcount4slurmarray = curr_job_config.get('jobcount4slurmarray', args.jobcount4slurmarray)
    slurmarraycmdfilepath = curr_job_config.get('slurmarraycmdfilepath', args.slurmarraycmdfilepath)
    code_snapshot_dir = os.environ["RSC_RUN_SLURM_SNAPSHOT_DIR"]

    cmd_to_append_to = f"{os.path.join(code_snapshot_dir, 'srun-wrapper-slurmarray.sh')}"

    srun_cmd_str = append_args_to_cmdline(
        curr_job_config,
        args,
        cmd_to_append_to,
    )

    # Things particular to slurmarray
    if slurmarraycmdfilepath != None:
        srun_cmd_str += f" --slurmarraycmdfilepath {slurmarraycmdfilepath}"

    sbatch_cmd, sbatch_cmd_str = gen_sbatch_command_and_str(curr_job_config, srun_cmd_str)

    if curr_job_config.get("sbatch_cmd_str", None) is None:
        curr_job_config["sbatch_cmd_str"] = sbatch_cmd_str

    # TODO: support MANUAL requeue and resume feature for slurm array
    # log_job_config(curr_job_config, exp_run_dir)

    print(f"srun cmd : {srun_cmd_str}")
    print(f"sbatch cmd: {sbatch_cmd}")

    print(f"SBATCH_CMD_STR : {sbatch_cmd_str}   @@   SBATCH_CMD : {sbatch_cmd}")

    if args.localnode == None:
        run_batch(env=os.environ.copy(), sbatch_cmd_str=sbatch_cmd_str, sbatch_cmd=sbatch_cmd)
    else:
        # pass environment variable to remotely run process
        envars=['RSC_AVATAR_METADATA_PATH', 'RSC_AVATAR_RSCASSET_PATH', 'RSC_AVATAR_READONLY_PATH', 'RSC_AVATAR_DEBUGDATA_PATH', 'RSC_AVATAR_EVAL_CONFIG_PATH', 'CONDA_PREFIX', 'LD_LIBRARY_PATH', 'RSC_EXP_RUN_BASE_DIR', 'RSC_RUN_SLURM_SNAPSHOT_DIR', 'RSC_UA_MINI_BATCH_SIZE', 'RSC_UA_MAX_ITER', 'RSC_UA_NUM_AIRSTORE_WORKERS', 'RSC_CONFIG_DIR', 'RSC_CONFIG_DIR', 'RSC_PROGRAM_NAME', 'RSC_JOB_RESUME_DIR', 'RSC_JOB_UUID']
        estring=""
        for var in envars:
            estring += "&& "
            estring += f"export {var}=${var} "

        # replace srun with remote ssh run
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
