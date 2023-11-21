"""
Tool that you can use to generate a bunch of commands to run on a slurm array in Avatar RSC
NOTE: Users are not supposed to call this directly, use `run-array.py` instead.
"""

import copy
import argparse
import datetime
import json
import platform
import os
import shlex
import shutil
import subprocess
import sys

from sbatch import (
    append_args_to_cmdline,
    gen_job_config,
    log_job_config,
    read_job_config,
    run_hash,
    run_dir,
    slurm_snapshot_code_dir,
    code_snapshot,
    get_parser as get_parser_sbatch
)

from typing import List, Optional

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
    requiredNamed.add_argument(
        "--slurmarraytaskid",
        type=int,
        default=-1,
        help="slurmarraytaskid",
        required=True
    )
    return parser


def get_args(input_args: Optional[List[str]] = None):
    """
    input_args (List[str]): strings to parse, defaults to sys.argv
    """
    parser = get_parser()
    args = parser.parse_args(input_args)
    return args


def main(scheduler_args: Optional[List[str]] = None):

    args = get_args(scheduler_args)

    # NOTE(we do not support resuming entire arrays)
    curr_job_config = gen_job_config(args)

    exp_hash = run_hash(args.slurmarraytaskid)
    exp_run_leaf_dir = run_dir(args, exp_hash)
    os.makedirs(args.checkpoint_root_dir, exist_ok=True)
    exp_run_dir = os.path.join(args.checkpoint_root_dir, exp_run_leaf_dir)
    os.makedirs(exp_run_dir, exist_ok=True)
    code_snapshot_dir = slurm_snapshot_code_dir(exp_run_dir)

    # Re-use sbatch's command building utils
    cmd_to_append_to = f"{os.path.join(code_snapshot_dir, 'srun-wrapper.sh')}"
    srun_cmd_str = append_args_to_cmdline(
        curr_job_config,
        args,
        cmd_to_append_to,
    )

    log_job_config(curr_job_config, exp_run_dir)

    # Append/create to command file
    wpath = args.slurmarraycmdfilepath

    # Find the part of the command that we want to write
    stoptoken = "srun-wrapper.sh"  # eliminate SLURM/SBATCH/ENV setup related arguments and write only application-specific arguments
    stoppos = srun_cmd_str.find(stoptoken) + len(stoptoken)
    chopped_srun_cmd = srun_cmd_str[stoppos:].strip()  # Chop blank chars

    wpath_exists = os.path.exists(wpath)
    open_mode = "a" if wpath_exists else "wt"
    with open(wpath, open_mode) as wfd:
        wfd.write(f"{chopped_srun_cmd}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
