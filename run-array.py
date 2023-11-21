"""
Tool that luanches an array of slurm jobs for you.
"""

import importlib
# NOTE(julieta) consider re-naming these modules with underscores instead of dashes, so we can import them plainly
sbatch_gen_cmd = importlib.import_module("sbatch-gen-cmd")

import tempfile
import argparse
import os
from typing import List, Optional
from datetime import datetime


USER = os.environ.get('USER')
IDS_IN_AIRSTORE = 70


def _load_id_projects() -> List[str]:
    """Load a list of id projects"""
    ids_fname = os.path.join(os.path.dirname(__file__), "ablation-config/projects.txt")
    # print(ids_fname)
    with open(ids_fname, "r")  as f:
        ids = f.readlines()
        assert len(ids) == IDS_IN_AIRSTORE, f"expected {IDS_IN_AIRSTORE} but found {len(ids)}"
    return ids


def build_preamble(user: str, latest_conda: str) -> str:
    """Build preamble code that makes sure conda is loaded and we are in the right directory"""
    activate_conda = f"source {latest_conda}/bin/activate"
    cd_to_dir      = f"cd /home/{user}/rsc/neurvol2-jason/"
    return " && ".join([activate_conda, cd_to_dir])


def build_rsc_launcher_cmd(nids: int, use_projects_file: bool = False) -> str:
    """Build the part of the command that loads rsc projects and calls the rsc launcher
    Args:
        nids: number of ids to load. Zero means none will be loaded.
        use_projects_file: Whether to create a file of projects instead.
    """
    project_ids = _load_id_projects()

    preamble = "SCENV=ava rsc_launcher launch"

    if nids == 0:
        out = f"{preamble} --no-projects"

    # Load projects and save them to a temp file
    elif use_projects_file:
        # Create small file to load projects from
        projects_path = os.path.join(os.path.dirname(__file__), f"ablation-config/projects{nids}.txt")
        with open(projects_path, "w") as f:
            f.writelines(project_ids[:nids])
        out = f"{preamble} --projects-file {os.path.abspath(projects_path)}"
    else:
        # Add all the projects to the command, might fail for very large number of projects
        project_ids = [x.strip() for x in project_ids][:nids]
        id_projects_list = " ".join(project_ids)
        out = f"{preamble} --projects {id_projects_list}"

    out += " --key-copy-concurrency 5 -e "
    return out


def build_main_cmd(
    cmd: str,
    nids: int,
    nnodes: int,
    ngpus: int,
    ntasks: int,
    nworkers: int,
    batch_size: int,
    checkpoint_root_dir: str,
    ablation_camera: bool,
    slurmarraycmdfilepath: str,
    learning_rate: float,
    slurmarraytaskid: Optional[int] = None,
    jobcount: Optional[int] = None,
) -> str:
    """Build the sbatch command
    Args:
        nnodes: number of nodes (1 node = 1 machine with up to 8 GPUs) to use
        ngpus: number of gpus per node
        ntasks: number of tasks per gpu TODO(julieta) find out what this is
        nworkers: number of airstore dataloader workers
    """

    pre  =           f"python3 {cmd} -n {nnodes} -g {ngpus} -t {ntasks} -w {nworkers}"
    source_dir =     f" --source-dir /home/{USER}/rsc/neurvol2-jason/"
    checkpoint_dir = f" --checkpoint-root-dir {checkpoint_root_dir}"
    idfile =         f" --idfilepath /checkpoint/avatar/giriman/idfiles/70ids-m0.txt"
    ablation_camera = " --ablationcamera" if ablation_camera else ""
    batch_size =     f" --batchsize {batch_size}"
    learning_rate =  f" --learning-rate {learning_rate}"
    rest =            " --displayloss"
    slurmarraycmdfilepath = f" --slurmarraycmdfilepath {slurmarraycmdfilepath}"
    slurmarraytaskid = f" --slurmarraytaskid {slurmarraytaskid}" if slurmarraytaskid is not None else ""
    jobcount4slurmarray = f" --jobcount4slurmarray {jobcount}" if jobcount is not None else ""

    return pre + \
        source_dir + \
        checkpoint_dir + \
        idfile + \
        ablation_camera + \
        batch_size + \
        learning_rate + \
        rest + \
        slurmarraycmdfilepath + \
        slurmarraytaskid + \
        jobcount4slurmarray


def main():
    parser = argparse.ArgumentParser(description="Build a command to run on RSC")
    parser.add_argument("--nids", "-n",      type=int, default=70, help="Number of ids to train on")
    parser.add_argument("--nnodes",          type=int, default=1)
    parser.add_argument("--ngpus",           type=int, default=8)
    parser.add_argument("--ntasks",          type=int, default=8)
    parser.add_argument("--nworkers",        type=int, default=4)
    parser.add_argument("--batchsize",       type=int, default=4)
    parser.add_argument("--learning-rate",   type=float, default=0.001, help="Learning rate")
    parser.add_argument("--ablation_camera", action="store_true", default=False, help="Whether to ablate cameras")
    parser.add_argument("--latest-conda", type=str, default="/checkpoint/avatar/conda-envs/ua-env-20220804-airstore-326/", help="latest conda environment to use to run experiments on RSC")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Don't actually run the commands")

    args = parser.parse_args()

    # TODO(julieta) get hdir from command line args
    hdir=f"/checkpoint/avatar/{USER}/array-cmds/array-cmd-" + datetime.now().isoformat().replace(":", "_")

    print(args)

    if not args.dry_run:
        os.system(f"ava mkdir -p {hdir}")

    # STEP1: generate commandline file and create output dirs in rsc
    print("Step 1")

    job_id = 0
    with tempfile.TemporaryDirectory() as tmpdirname:
        for i, bsz in enumerate([4, 8]):
            sbatch_cmd = build_main_cmd(
                "sbatch-gen-cmd.py",
                args.nids,
                args.nnodes,
                args.ngpus,
                args.ntasks,
                args.nworkers,
                bsz,
                tmpdirname,
                args.ablation_camera,
                f"{tmpdirname}/commands.txt",  # slurmarraycmdfilepath,
                args.learning_rate,
                slurmarraytaskid=i,
            )

            # Remove 'python3' and module invocations
            sbatch_cmd = sbatch_cmd.split(" ")[2:]

            # Simply call the module locally to generate and save the command
            sbatch_gen_cmd.main(sbatch_cmd)

            job_id += 1

        # Copy over to avarsc before the temp dir gets cleaned up
        if not args.dry_run:
            os.system(f"ava sync {tmpdirname}/ :{hdir}")

    # STEP2: launch slurmarray command
    outer_cmd = build_rsc_launcher_cmd(args.nids)
    inner_cmd = build_preamble(USER, args.latest_conda)
    sbatch_cmd = build_main_cmd(
        "sbatch-slurmarray.py",
        args.nids,
        args.nnodes,
        args.ngpus,
        args.ntasks,
        args.nworkers,
        args.batchsize,
        hdir,
        args.ablation_camera,
        f"{hdir}/commands.txt",
        args.learning_rate,
        slurmarraytaskid=None,
        jobcount=job_id,
    )
    array_cmd = outer_cmd + "'" + inner_cmd + " && " + sbatch_cmd + "'"

    print("Step 2")
    print(array_cmd)
    if not args.dry_run:
        os.system(array_cmd)


if __name__ == "__main__":
    main()
