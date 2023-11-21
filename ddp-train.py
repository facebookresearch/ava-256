"""Train an autoencoder."""
import argparse
import binascii
import gc
import hashlib as hlib
import importlib
import importlib.util
import itertools
import json
import logging
import math
import numpy as np
import os
import pickle
import platform
import random
import subprocess
import sys
import torch
import torch.distributed as dist
import torch.utils.data
import time
import psutil
import pathlib

import pandas as pd

from lion_pytorch import Lion
from torch.utils.tensorboard import SummaryWriter
from subprocess import CalledProcessError

from data.mugsy_dataset import MugsyCapture
from data.mugsy_dataset import MultiCaptureDataset as MugsyMultiCaptureDataset
from data.mugsy_dataset import none_collate_fn

from models.volumetric_multi3 import (
    AETIME,
    IDENCTIME,
    ENCTIME,
    DECTIME,
    RAYMARCHINGTIME,
    VERTLOSSTIME,
    RGBLOSSTIME,
    COLORCALANDBGTIME,
)


sys.dont_write_bytecode = True

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

# TODO(julieta) see if this actually does anything
torch.backends.cudnn.benchmark = True # gotta go fast!


def display_cudamem(msg, devcnt):
    dmsg = ""
    for i in range(devcnt):
        mem = torch.cuda.max_memory_allocated(f"cuda:{i}")
        rmem = torch.cuda.max_memory_reserved(f"cuda:{i}")

        dmsg += str(mem)
        dmsg += ':'
        dmsg += str(rmem)
        dmsg += ','
        # note that
        logging.info("CUDA {} PEAK MEM: {} Reserved MEM: {} -- acc msg for all assigned GPUs: {}".format(i, mem, rmem, dmsg))
        #logging.info("CUDA MEM DETAIL FOR CUDA 0 : {}".format(torch.cuda.memory_stats("cuda:0")))

def run_shell_command(command: str, timeout: int) -> subprocess.CompletedProcess:
    retval = None
    try:
        retval = subprocess.run([command],
                                shell=True,
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=timeout)
    except subprocess.TimeoutExpired as e1:
        logger.error(str(e1))
        logger.error("Timeout raised, from command {}".format(str(command)))
    except subprocess.SubprocessError as e2:
        logger.error(str(e2))
        logger.error("SubprocessError raised from command {}".format(str(command)))
    except Exception as e3:
        logger.error(str(e3))
        logger.error("Exception raised from command {}".format(str(command)))
    return retval

def log_kerberos_ticket_details(iternum, rank):
    try:
        retval = run_shell_command(command=KLIST_CMD, timeout=KLIST_CMD_TIMEOUT_SECONDS)
        logging.debug("klist retval: %s", str(retval))
        try:
            retval.check_returncode()
        except CalledProcessError as e1:
            logging.error(str(e1))
            logging.error("klist failed, CalledProcessError raised...")
            return
        logging.debug(retval.stdout)
        logging.debug(retval.stderr)
        klist_output_stdout = retval.stdout.decode("utf-8")
        klist_output_stderr = retval.stderr.decode("utf-8")
        logging.info(f"Rank: {rank} hostname: {platform.node()} Iternum: {iternum} klist output(stdout): {klist_output_stdout}")
        logging.info(f"Rank: {rank} hostname: {platform.node()} Iternum: {iternum} klist output(stderr): {klist_output_stderr}")
    except Exception as e:
        logging.error(str(e))
        logging.error("Exception raised while processing sinfo stdout")
        return

def gen_optimizer(
    net: torch.nn.Module,
    optim_type: str,
    batchsize: int,
    rank: int,
    learning_rate: float,
    # enableddp: bool,
    tensorboard_logger=None,
    encoder_lr=None,
):
    # optim_type = os.environ.get("RSC_UA_OPTIMIZER", "adam")
    # if enableddp:
    #     # for distributed training
    #     print(" gen optimizer: enableddp : {}".format(enableddp))
    #     net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[0])
    if encoder_lr is not None:
        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
            cur_net = net.module
        else:
            cur_net = net
        print(f"type(cur_net): {type(cur_net)}")
        encoder_params = list(filter(lambda x: x.requires_grad,itertools.chain([v for k, v in cur_net.named_parameters() if k.startswith("encoder")])))
        other_params = list(filter(lambda x: x.requires_grad,itertools.chain([v for k, v in cur_net.named_parameters() if not k.startswith("encoder")])))
        params = [{"params": encoder_params, "lr": encoder_lr}, {"params": other_params}]
    else:
        params = filter(lambda x: x.requires_grad,itertools.chain(net.parameters()))
    if optim_type == "adam":
        opt = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))
    elif optim_type == "sgd":
        opt = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    elif optim_type == "adamw":
        opt = torch.optim.AdamW(params, lr=learning_rate, betas=(0.9, 0.999))
    elif optim_type == "dadapt_adam":
        opt = DAdaptAdam(params, lr=1.0, betas=(0.9, 0.999))
    elif optim_type == "lion":
        opt = Lion(params, lr=learning_rate)

    if tensorboard_logger and rank == 0:
        worldsize = int(os.environ.get("SLURM_NTASKS"))
        tb_hyperparams = {
            "minibatchsize": batchsize,
            "globalbatchsize": batchsize * worldsize,
            "learningrate": learning_rate,
            "optimizer": optim_type,
            }
        tensorboard_logger.add_hparams(tb_hyperparams, {"hp_metric": 1.0}, run_name=".")
    return params, opt

def import_module(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def tocuda(d):
    if isinstance(d, torch.Tensor):
        return d.to("cuda")
    elif isinstance(d, dict):
        return {k: tocuda(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [tocuda(v) for v in d]
    else:
        return d

def findbatchsize(d):
    if isinstance(d, torch.Tensor):
        return d.size(0)
    elif isinstance(d, dict):
        return findbatchsize(next(iter(d.values())))
    elif isinstance(d, list):
        return findbatchsize(next(iter(d.values())))
    else:
        return None

def check_quorum(token, allmembers):
    size = dist.get_world_size()
    dist.all_reduce(token, op=dist.ReduceOp.SUM, group=allmembers)
    ResetFlag = False
    if token[0] !=  size:
        ResetFlag = True
        # logging.warning(f"at least one of the ranks is seeing unstable loss value(s)")
    return ResetFlag

def save_meta(path, mds):
    logging.info(" save meta : start ... ")
    # datasetmulti:
    dm=dict()
    dm['identities'] = mds.identities
    dm['camera_names'] = mds.camera_names
    dm['camera_maps'] = mds.camera_maps
    dm['nsamples'] = mds.nsamples
    dm['frame_sets'] = mds.frame_sets

    # if meta is saved by genesis_multi_division_ghsv2_pit, golbal texmean/std/vertmean/std are NONE
    # CALL calc_tex_stats, calc_vert_stats to get golbal stats
    dm['texmean'] = mds.texmean
    dm['texstd'] = mds.texstd
    dm['vertmean'] = mds.vertmean
    dm['vertstd'] = mds.vertstd

    # if meta is saved by genesis_multi_division_ghsv2_pit, each ID keeps its own texmean/std/vertmean/std values
    dm['datasets'] = mds.datasets

    dm['img_size'] = mds.img_size
    dm['identities_str'] = mds.identities_str # identity in string type

    opath = path+'/dataset_meta.npy'
    np.save(opath, dm, allow_pickle=True)

    logging.info(" save meta : done ")


def garbage_collection(iternum, rank):
    import gc
    gc_start = round(time.time() * 1000)
    gc.collect()
    gc_duration = round(time.time() * 1000) - gc_start
    logging.debug(f"Rank {rank}: garbage collection at iteration {iternum} took {gc_duration} ms")

def reset_perf_stats(perf_stats):
    perf_stats["acc_iter_time"] = 0
    perf_stats["acc_dataload_time"] = 0
    perf_stats["acc_forward_time"] = 0
    perf_stats["acc_ae_time"] = 0
    perf_stats["acc_backward_time"] = 0
    perf_stats["acc_postiter_time"] = 0
    perf_stats["acc_over_iterations"] = 0

def log_perf_stats(iternum, rank, perf_stats):
    avg_iter_time = round(perf_stats['acc_iter_time'] / perf_stats['acc_over_iterations'], 3)
    avg_dataload_time = round(perf_stats['acc_dataload_time'] / perf_stats['acc_over_iterations'], 3)
    avg_forward_time = round(perf_stats['acc_forward_time'] / perf_stats['acc_over_iterations'], 3)
    avg_ae_time = round(perf_stats['acc_ae_time'] / perf_stats['acc_over_iterations'], 3)
    avg_backward_time = round(perf_stats['acc_backward_time'] / perf_stats['acc_over_iterations'], 3)
    avg_postiter_time = round(perf_stats['acc_postiter_time'] / perf_stats['acc_over_iterations'], 3)
    logging.info(f"Rank {rank} "
        f"Iteration {iternum} "
        f"avg_iter_time {avg_iter_time} "
        f"avg_dataload_time {avg_dataload_time} "
        f"avg_forward_time {avg_forward_time} "
        f"avg_ae_time {avg_ae_time} "
        f"avg_backward_time {avg_backward_time} "
        f"avg_postiter_time {avg_postiter_time} "
        f"avg_over_iterations {perf_stats['acc_over_iterations']} "
        f"hostname {platform.node()} ")

    reset_perf_stats(perf_stats)


if __name__ == "__main__":
    __spec__ = None # to use ipdb

    # parse arguments
    logging.info(" TRAIN PROC : OS PID  {}".format(os.getpid()))
    parser = argparse.ArgumentParser(description='Train an autoencoder')
    parser.add_argument('experconfig', type=str, help='experiment config file')
    parser.add_argument('--profile', type=str, default="Train", help='config profile')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parser.add_argument('--ablationcamera', action='store_true', help='running ablation camera experiments')
    parser.add_argument('--ablationcamerasinglegpu', action='store_true', help='running ablation camera experiments on single gpu')
    parser.add_argument('--debugprefetch', action='store_true', help='prefetch N batches in CPU memory and no storage access during training')
    parser.add_argument('--noprogress', action='store_true', help='don\'t output training progress images')
    parser.add_argument('--nostab', action='store_true', help='don\'t check loss stability')
    parser.add_argument('--rank', type=int, default=0,  help='process rank in distributed training')
    parser.add_argument('--worldsize', type=int, default=1,  help='the number of processes for distributed training')
    parser.add_argument('--masterip', type=str, default="localhost", help='master node ip address')
    parser.add_argument('--masterport', type=str, default="43321", help='master node network port')
    parser.add_argument('--holdoutpath', type=str, default=None, help='directory to holdout info')
    parser.add_argument('--holdoutratio', type=str, default=None, help='cam hold out ratio')
    parser.add_argument('--disableddp', action='store_true', help='don\'t check loss stability')
    parser.add_argument('--displayloss', action='store_true', help='logging loss value every iteartion')
    parser.add_argument('--disableshuffle', action='store_true', help='no shuffle in airstore')
    parser.add_argument('--shard_air', action='store_true', help='no shuffle in airstore')
    parser.add_argument('--seed_air', type=str, default=None, help='seed value for airstore shuffling. enable determinisitc should be set to True together')
    parser.add_argument('--enabledeterministic', action='store_true', help='fixed seed')
    parser.add_argument('--evalcheckpointpath', type=str, default=None, help='checkpoint file path for evaluating ablation experiments')
    parser.add_argument('--ids2use', type=int, default=-1,  help='the number of processes for distributed training')
    parser.add_argument('--idfilepath', type=str, default=None, help='file of id list for training or evaluation')
    parser.add_argument('--evaldatapath', type=str, default=None, help='path evaluation data from PIT') # /checkpoint/avatar/jinkyuk/read-only/ablation-eval/bz2/[path]

    # Logging options
    parser.add_argument('--displayprofstats', action='store_true', help='logging perf stats every iteartion. Will make training slower')
    parser.add_argument('--tensorboard-logdir', type=str, default=None, help='dir path where tensorboard log will output to')
    parser.add_argument('--logallrankstb', action='store_true', help='log from all ranks to tb, not just rank 0. Can make tb slow/unusable for jobs with many GPUs')

    # Training hyperparams.
    # NOTE(julieta) These values can now sometimes come from 3 places: command line, env vars, or experiment config file
    # Everything else being equal, we respect the following order:
    # command line args have precedence over env vars, which have precedence over config file.
    # In this script, we are trying to move away from environment variables to make it easier to do sweeps with slurm,
    # but the launchers are still setting the env variables because other scripts (eg eval) may rely on them.
    parser.add_argument('--dataset', type=str, default="MGR", help='dataset to train on, mugsy or mgr')
    parser.add_argument('--maxiter', type=int, default=10_000_000, help='maximum number of iterations')
    parser.add_argument('--batchsize', type=int, default=4, help='Batch size per GPU to use')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate passed as it to the optimizer')
    parser.add_argument('--clip', type=float, default=1., help='Gradient clipping')
    parser.add_argument('--nids', type=int, default=1, help='Number of identities to select')
    parser.add_argument('--subsample-size', type=int, default=1134 // 4, help="Size of image after cropping -- or other subsampling method used")
    parser.add_argument('--downsample', type=int, default=4 , help="image downsampling factor at data loader -- default 4, DL return images with H=1024, W=667")
    parser.add_argument("--disable_id_encoder", action='store_true', help="disable id_encoder in ae")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="steps to accumulate gradients before updating model weights")
    parser.add_argument("--encoder_lr", type=float, default=None, help="learning rate for encoder ",)
    parser.add_argument("--encoder_channel_mult", type=int, default=1, help="channel multiplier for the encoder ",)

    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()

    logallrankstb = bool(args.logallrankstb)
    enableddp = not args.disableddp

    # TODO(julieta) remove all references to SLURM variables
    args.worldsize = int(os.environ.get("SLURM_NTASKS", 1))
    args.rank = int(os.environ.get("SLURM_PROCID", 0))

    outpath = os.environ.get('RSC_RUN_DIR', os.path.abspath(__file__)) # RSC_EXP_RUN_BASE_DIR/SLURM_NODEID/SLURM_LOCALID

    tlogpath = "{}/log-r{}.txt".format(outpath, args.rank)

    # if path exists append automatically
    logging.info(f"Python {sys.version}")
    logging.info(f"PyTorch {torch.__version__}")
    logging.info(" ".join(sys.argv))
    logging.info(f"Output path: {outpath}")

    logging.info(" ==================================================== ")
    logging.info(" args.rank: {}, args: {}".format(args.rank, args))
    v_devices = os.getenv('CUDA_VISIBLE_DEVICES', 'NOTFOUND')
    logging.info(" ==================================================== ")
    logging.info(" At rank {} DEVICE COUNT : {} -- CUDA VISIBLE DEVICES {}" .format(args.rank, torch.cuda.device_count(), v_devices))
    logging.info(" ==================================================== ")

    cuda_device_count = torch.cuda.device_count()

    tensorboard_logger = None
    if args.tensorboard_logdir is not None:
        if logallrankstb or args.rank == 0:
            tensorboard_logdir = os.path.join(
                args.tensorboard_logdir,
                os.environ['SLURM_JOB_ID'],
                os.environ['SLURM_PROCID'],
            )
            logging.info(f"Creating tensorboard output at {tensorboard_logdir}")
            tensorboard_logger = SummaryWriter(tensorboard_logdir)

    logging.info("@@@@@@@@@@@@@@@ JOB CONFIG:")
    logging.info(args)
    logging.info("@@@@@@@@@@@@@@@ END OF JOB CONFIG \n")

    logging.info("@@@@@@@@@@@@@@@ OS ENV VARIABLES (from sbatch):")
    for k, v in os.environ.items():
        logging.info(f"{k}:{v}")
    logging.info("@@@@@@@@@@@@@@@ END OF OS ENV VARIABLES :")

    seed_air_hash = None
    # convert string seed arg into numeric hash value and cut them into 8 digit for c++ airstore client library
    if args.seed_air:
        signature = args.seed_air
        val=hlib.md5(signature.encode('utf-8'))  # make a persistent md5 string for SID+MCD+MTIME
        lval = int(val.hexdigest(), 16)
        seed_air_hash = str(lval)[:9]
        logging.info("CONVERT USER SEED STRING {} into numeric value {}".format(args.seed_air, seed_air_hash))
    else:
        logging.info(" NO USER SEED VALUE PROVIDED")

    if args.ablationcamera or args.ablationcamerasinglegpu:
        assert(args.holdoutpath != None)
        assert(args.holdoutratio != None)
        logging.info("[ABLATIONCAMERA: holdoutpath {} holdoutratio {}".format(args.holdoutpath, args.holdoutratio))
    else:
        logging.info("[NO ABLATION TEST]")

    #args.masterip = os.environ.get("SLURM_SUBMIT_HOST")
    slurm_job_nodes_tmp = os.environ.get("SLURM_JOB_NODELIST") # string type: rsclearnXXXX or rsclearn[XXXX-YYY,...]
    logging.info(f"slurm_job_nodes_tmp: type {type(slurm_job_nodes_tmp)}, content {slurm_job_nodes_tmp}")

    # master_node = find_masternode(slurm_job_nodes_tmp)
    # logging.info(f"slurm_job_nodes: {master_node}")
    # args.masterip = master_node

    logging.info(f"master ip {args.masterip}")

    rank = args.rank
    worldsize = args.worldsize

    disturl=f"tcp://{args.masterip}:{args.masterport}"
    logging.info(f" DIST URL : {disturl}")

    # TODO(julieta) get the number of workers from the command line
    numworker = 4

    if enableddp or args.worldsize > 1:
        # Start a ddp group even if ddp is "disabled" because we want to be able to average losses across all ranks
        logging.info(" INIT DDP RANK {}  WSIZE {}  URL {}".format(args.rank, args.worldsize, disturl))
        dist.init_process_group(backend='nccl', init_method=disturl, world_size=args.worldsize, rank=args.rank)
        logging.info(" distributed training group is initialized at rank : {}".format(args.rank))

    allmembers = None
    loss_quorum = None
    if enableddp:
        allmembers = dist.new_group(range(worldsize))
        loss_quorum = torch.tensor([1]).cuda()

    # load config
    starttime = time.time()
    experconfig = import_module(args.experconfig, "config")
    unparsed_args = {k: v for k, v in vars(args).items() if k not in parsed}
    logging.info(f"Unparsed args: {unparsed_args}")
    profile = getattr(experconfig, args.profile)(**unparsed_args)
    if not args.noprogress:
        progressprof = experconfig.Progress()

    logging.info("@@@@@@@ CONFIG.PY PATH : {}".format(experconfig))

    # Load batchsize config; override with env var; override with command line arg
    # TODO(julieta) Get batch size from cli
    batchsize = args.batchsize
    # batchsize = profile.batchsize
    # if os.environ.get("RSC_UA_MINI_BATCH_SIZE", None):
    #     batchsize = int(os.environ.get("RSC_UA_MINI_BATCH_SIZE"))
    # if args.batchsize is not None:
    #     batchsize = args.batchsize

    learning_rate = args.learning_rate

    logging.info("Config loaded ({:.2f} s)".format(time.time() - starttime))

    # build dataset & testing dataset
    starttime = time.time()

    # TODO: clean up on PR
    logging.info("@@@@@@ image related configs: subsample_size {}, downsample {}".format(args.subsample_size, args.downsample))

    # Load
    nr_captures = pd.read_csv(pathlib.Path(__file__).parent / "215_ids.csv", dtype=str)
    nr_captures = [MugsyCapture(mcd=row['mcd'], mct=row['mct'], sid=row['sid'], is_relightable=False) for _, row in nr_captures.iterrows()]

    r_captures =  pd.read_csv(pathlib.Path(__file__).parent / "345_ids.csv", dtype=str)
    r_captures =  [MugsyCapture(mcd=row['mcd'], mct=row['mct'], sid=row['sid'], is_relightable=True) for _, row in r_captures.iterrows()]

    captures = nr_captures + r_captures

    train_captures = captures[:args.nids]
    train_captures = np.array_split(train_captures, worldsize)[args.rank]
    dataset = MugsyMultiCaptureDataset(train_captures, downsample=args.downsample)

    logging.info("DS SET UP IS DONE  -- number of workers : {} -- using configfile ings ".format(numworker))


    logging.info(" DL argument : enable_deterministic {}, disable_shuffle_air : {}".format(args.enabledeterministic, args.disableshuffle))
    #torch.utils.data.DataLoader
    #dataloader = airdl.DataLoader(dataset,
    logging.info("[dl-refactor] dataloader created with torch.utils.data.DataLoader class")

    evalpoints = list()
    maxiter = args.maxiter  # that ought to be enough for anyone

    for i in range(maxiter//200):
        evalpoints.append(i*200)

    logging.info(" maxiter :  {}, batchsize: {}".format(maxiter, batchsize))

    #dummysampler = AirstoreDummySampler(dataset)
    dummysampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=maxiter*batchsize*2) # infinite random sampler


    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batchsize,
                                             sampler=dummysampler,
                                             shuffle=False,
                                             drop_last=True,
                                             num_workers=numworker,
                                             pin_memory=True,
                                             collate_fn=none_collate_fn,
                                            )

    logging.info("Dataset instantiated ({:.2f} s)".format(time.time() - starttime))

    # data writer
    starttime = time.time()
    if not args.noprogress:
        writer = progressprof.get_writer()
        logging.debug("Writer instantiated ({:.2f} s)".format(time.time() - starttime))

    # build autoencoder
    starttime = time.time()
    ae = profile.get_autoencoder(dataset, args.disable_id_encoder, args.encoder_channel_mult)
    ae = ae.to("cuda").train()

    iternum = 0

    if enableddp:
        # TODO(julieta) control whether we want to distribute the full model, or just a subset
        # ae = torch.nn.parallel.DistributedDataParallel(ae, device_ids=[0], find_unused_parameters=False)
        ae.encoder = torch.nn.parallel.DistributedDataParallel(ae.encoder, device_ids=[0], find_unused_parameters=False)

    optim_type = "adam"
    _, optim = gen_optimizer(ae, optim_type, batchsize, args.rank, learning_rate, tensorboard_logger, encoder_lr=args.encoder_lr)

    logging.info("Autoencoder instantiated ({:.2f} s)".format(time.time() - starttime))

    starttime = time.time()

    lossweights = profile.get_loss_weights()
    logging.info("Optimizer instantiated ({:.2f} s)".format(time.time() - starttime))

    # train
    starttime = time.time()
    prevloss = np.inf

    prefetch_cpu=list()
    if args.debugprefetch == True:
        debug_path = os.getenv('RSC_AVATAR_DEBUGDATA_PATH')
        file_found = True
        for k in range(nbatch_prefetch):
            fn = os.path.join(debug_path, f"prefetch-bsize{batchsize}-batch_{k}.npy")
            if not os.path.exists(fn):
                file_found = False
                break

        if file_found == False:
            pdl = iter(dataloader)
            for ii in range(nbatch_prefetch):
                data = next(pdl)
                prefetch_cpu.append(tocpu(data))
                if ii % 10 == 0 :
                    logging.info(f"\t @@ prefetching TO CPU MEMORY progress {ii} out of {nbatch_prefetch}   at rank {args.rank}")

            assert(nbatch_prefetch == len(prefetch_cpu))
            logging.info(f" PREFETCH TO CPU MEMORY IS DONE at {args.rank}")

            if True:
                for k in range(nbatch_prefetch):
                    np.save(debug_path+f"/prefetch-bsize6-batch_{k}.npy", prefetch_cpu[k], allow_pickle=True)
                    logging.info(f" SAVE - {k}")
        else:
            for k in range(nbatch_prefetch):
                idx = k % nbatch_prefetch # assume that there are 100 pre saved batches
                buf = np.load(debug_path+f"/prefetch-bsize6-batch_{idx}.npy", allow_pickle=True).tolist()
                prefetch_cpu.append(buf)
                logging.info(f" reload - {k}")
            logging.info(f"RELOADING PRE SAVED BATCHED TO CPU MEMORY  -- done at rank {args.rank}")

    fetchid = 0
    lstart = time.time()

    prevtime = time.time()

    perf_stats = {}
    if args.displayprofstats:
        reset_perf_stats(perf_stats)

    garbage_collection(0, args.rank)
    log_kerberos_ticket_details(0, args.rank)

    outputlist = profile.get_outputlist() if hasattr(profile, "get_outputlist") else []
    if args.displayprofstats:
        outputlist += [
            AETIME,
            IDENCTIME,
            ENCTIME,
            DECTIME,
            RAYMARCHINGTIME,
            VERTLOSSTIME,
            RGBLOSSTIME,
            COLORCALANDBGTIME,
        ]

    # NOTE(julieta) We use this list exclusively for "in" tests, so a set is more fitting. Consider changing the name
    outputlist = set(outputlist)
    logging.info(" OUTPUT LIST :{}".format(outputlist))

    # self.bpath = bpath
    # self.hpath = hpath
    # self.spath = spath
    # self.fpath = fpath

    #assert(len(dataset.datasets) == 1)

    if args.holdoutpath != None and args.ablationcamera == True:
        logging.info("ABLATION EXPERIMENT")
        fn = "{}/sampleseq_{}.txt".format(os.environ['RSC_RUN_DIR'], args.rank)
        wfd = open(fn, 'wt')
        wfd.write(f"HEADER:START, args.rank:{args.rank}, args.holdoutpath:{args.holdoutpath}, args.holdoutratio:{args.holdoutratio}\n")
        for ds in dataset.datasets:
            wfd.write(f"sid {ds.sid} ds.holdoutpath: {ds.holdoutpath}\n")
            wfd.write(f"bpath:{ds.bpath}, hpath:{ds.hpath}, spath:{ds.spath}, fpath:{ds.fpath}\n")
            wfd.write(f"ds.holdout[cam] : {ds.holdout['holdcams']}\n")
            wfd.write(f"frame_to_scams: fr count {len(ds.frame_to_scams.keys())}\n")
            for k, v in ds.frame_to_scams.items():
                wfd.write(f"fr {k}:{v}\n")
                # self.holdout = np.load(hpath, allow_pickle=True).tolist()
                # self.seginfo = np.load(spath, allow_pickle=True).tolist()
                # self.frame_to_scams = np.load(fpath, allow_pickle=True).tolist()
                # self.frames_to_exclude=list()
            wfd.write(f"frames_to_exclude (holdout frames): {ds.frames_to_exclude}\n")
        wfd.write("HEADER:END\n")
        wfd.flush
    else:
        logging.info("NO ABLATION EXPERIMENT")

    # while True:
    for data in dataloader:

        if data is None:
            continue

        iter_start_time = time.time()
        if args.displayprofstats:
            iter_total_time = 0
            iter_dataload_time = 0
            iter_forward_time = 0
            iter_backward_time = 0
            iter_postiter_time = 0

        # data = None

        cudadata = None
        if args.debugprefetch == True:
            data = prefetch_cpu[fetchid]
            cudadata = tocuda(data)
            fetchid = fetchid + 1
            fetchid = fetchid % nbatch_prefetch
        else:
            # data = next(diter)

            m1 = data['frameid']
            m2= data['cameraid']
            #m3 = data['subject_id']

            if args.holdoutpath != None and args.ablationcamera == True:
                wfd.write(f"{iternum}, {m1}, {m2}\n")
                if iternum % 100 == 0:
                    wfd.flush()

            cudadata = tocuda(data)

        if args.displayprofstats:
            torch.cuda.synchronize()
            iter_dataload_time = time.time() - iter_start_time
            perf_stats["acc_dataload_time"] += iter_dataload_time

        if args.displayprofstats:
            forward_start_time = time.time()


        if iternum % args.gradient_accumulation == 0:
            fw_profile = False
            if fw_profile:
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    output, losses = ae(
                        trainiter=iternum,
                        outputlist=outputlist,
                        losslist=lossweights.keys(),
                        **cudadata,
                        **(profile.get_ae_args() if hasattr(profile, "get_ae_args") else {}))

                logging.info(prof.key_averages().table(sort_by="cuda_time_total"))

            else:
                output, losses = ae(
                    trainiter=iternum,
                    outputlist=outputlist,
                    losslist=lossweights.keys(),
                    **cudadata,
                    **(profile.get_ae_args() if hasattr(profile, "get_ae_args") else {}))


            if args.displayprofstats:
                torch.cuda.synchronize()
                iter_forward_time = time.time() - forward_start_time
                perf_stats["acc_forward_time"] += iter_forward_time
                if AETIME in outputlist:
                    perf_stats["acc_ae_time"] += output[AETIME]

                iter_loss_sum_start_time = time.time()

            # compute final loss
            loss = sum([
                lossweights[k] * (torch.sum(v[0]) / torch.sum(v[1]).clamp(min=1e-6) if isinstance(v, tuple) else torch.mean(v))
                for k, v in losses.items()])

            if args.displayprofstats:
                torch.cuda.synchronize()
                # TODO(julieta) accumulate?
                iter_loss_sum_time = time.time() - iter_loss_sum_start_time
                iter_param_zeroing_start_time = time.time()

            # to avoid unsed parameter error during distributed gradient aggregation
            # for param in ae.parameters():
            #     loss = loss + 0.0*param.sum()

            if args.displayprofstats:
                torch.cuda.synchronize()
                # TODO(julieta) accumulate?
                iter_param_zeroing_time = time.time() - iter_param_zeroing_start_time

            with torch.no_grad():
                mse_img = torch.mean((output['sampledimg'] / 255. - output['irgbrec'].clamp(0., 255.) / 255.)**2, dim=1, keepdim=True)
                mse = torch.sum(mse_img * cudadata['imagemask']) / cudadata['imagemask'].sum()
                psnr = -10 * np.log10(mse.cpu().item())

            # if args.displayloss:
            #     logging.info("Rank {} Iteration {} loss = {:.5f}, ".format(args.rank, iternum, float(loss.item())) +
            #             ", ".join(["{} = {:.5f}".format(k,
            #                 float(torch.sum(v[0]) / torch.sum(v[1].clamp(min=1e-6)) if isinstance(v, tuple) else torch.mean(v)))
            #                 for k, v in losses.items()]))

            if args.displayprofstats:
                torch.cuda.synchronize()
                backward_start_time = time.time()

            local_explosion = not args.nostab and (loss.item() > 400 * prevloss or not np.isfinite(loss.item()))
            loss_explosion = False
            if enableddp:
                loss_quorum[0]=1
                if local_explosion:
                    loss_quorum[0]=0
                    logging.warning(f"rank {args.rank} is seeing unstable loss value(s): {loss.item()}")
                loss_explosion = check_quorum(loss_quorum, allmembers)
            else:
                if local_explosion:
                    logging.warning(f"rank {args.rank} is seeing unstable loss value(s): {loss.item()}")
                    loss_explosion = True

            if local_explosion:
                # raise RuntimeError("Cannot recover from loss explosion!")
                try:
                    logging.warning(f"Rank {args.rank} found a sample with loss explosion for id {cudadata['idindex'].tolist()}, setting no grad")
                except Exception as e:
                    print(e)
                # loss.register_hook(lambda grad: torch.zeros_like(grad))
                # loss_explosion = False

            loss.backward()
            params = [p for pg in optim.param_groups for p in pg["params"]]
            # if local_explosion:
            #     for p in params:
            #         if hasattr(p, "grad") and p.grad is not None:
            #             p.grad.data[:] = 0
            #             p.grad.data[:] = 0
            # else:
            for p in params:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad.data[torch.isnan(p.grad.data)] = 0
                    p.grad.data[torch.isinf(p.grad.data)] = 0

            torch.nn.utils.clip_grad_norm_(ae.encoder.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(ae.decoder.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(ae.colorcal.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(ae.bgmodel.parameters(), args.clip)
            optim.step()
            optim.zero_grad()
        else:
            with ae.encoder.no_sync():
                fw_profile = False
                if fw_profile:
                    with torch.autograd.profiler.profile(use_cuda=True) as prof:
                        output, losses = ae(
                            trainiter=iternum,
                            outputlist=outputlist,
                            losslist=lossweights.keys(),
                            **cudadata,
                            **(profile.get_ae_args() if hasattr(profile, "get_ae_args") else {}))

                    logging.info(prof.key_averages().table(sort_by="cuda_time_total"))

                else:
                    output, losses = ae(
                        trainiter=iternum,
                        outputlist=outputlist,
                        losslist=lossweights.keys(),
                        **cudadata,
                        **(profile.get_ae_args() if hasattr(profile, "get_ae_args") else {}))


                if args.displayprofstats:
                    torch.cuda.synchronize()
                    iter_forward_time = time.time() - forward_start_time
                    perf_stats["acc_forward_time"] += iter_forward_time
                    if AETIME in outputlist:
                        perf_stats["acc_ae_time"] += output[AETIME]

                    iter_loss_sum_start_time = time.time()

                # compute final loss
                loss = sum([
                    lossweights[k] * (torch.sum(v[0]) / torch.sum(v[1]).clamp(min=1e-6) if isinstance(v, tuple) else torch.mean(v))
                    for k, v in losses.items()])

                if args.displayprofstats:
                    torch.cuda.synchronize()
                    # TODO(julieta) accumulate?
                    iter_loss_sum_time = time.time() - iter_loss_sum_start_time
                    iter_param_zeroing_start_time = time.time()

                if args.displayprofstats:
                    torch.cuda.synchronize()
                    # TODO(julieta) accumulate?
                    iter_param_zeroing_time = time.time() - iter_param_zeroing_start_time

                with torch.no_grad():
                    mse_img = torch.mean((output['sampledimg'] / 255. - output['irgbrec'].clamp(0., 255.) / 255.)**2, dim=1, keepdim=True)
                    mse = torch.sum(mse_img * cudadata['imagemask']) / cudadata['imagemask'].sum()
                    psnr = -10 * np.log10(mse.cpu().item())


                if args.displayprofstats:
                    torch.cuda.synchronize()
                    backward_start_time = time.time()

                local_explosion = not args.nostab and (loss.item() > 400 * prevloss or not np.isfinite(loss.item()))
                loss_explosion = False

                loss.backward()

        # Compute iter total time anyway -- no extra syncing needed, there is already an implicity sync during `backward`
        # and an explicit one to check for loss explosion
        iter_total_time = time.time() - iter_start_time
        if args.displayprofstats:
            torch.cuda.synchronize()
            iter_backward_time = time.time() - backward_start_time
            perf_stats["acc_backward_time"] += iter_backward_time
            perf_stats["acc_iter_time"] += iter_total_time
            postiter_start_time = time.time()

        #print progress
        if iternum < 10000:
            if iternum % 100 == 0: #  and args.rank == 0 :
                writer.batch(iternum, iternum * batchsize + torch.arange(0), f"{outpath}/progress_{iternum}.jpg", **cudadata, **output)
        else:
            if iternum % 1000 == 0: # and args.rank == 0 :
                writer.batch(iternum, iternum * batchsize + torch.arange(0), f"{outpath}/progress_{iternum}.jpg", **cudadata, **output)

        # compute evaluation output
        #if not args.noprogress and iternum in evalpoints and args.rank == 0:
        #    writer.batch(iternum, iternum * batchsize + torch.arange(0), **cudadata, **output)

        save_checkpoints = False
        if iternum < 10_000:
            # to account for early loss explosions
            if iternum % 2_000 == 0:
                save_checkpoints = True
        else:
            if iternum % 20_000 == 0:
                save_checkpoints = True

        # save intermediate results only if rank 0 does not have loss explosion
        if save_checkpoints:
            # in case of ddp loss_explosion is true if any of the ranks have loss explosion
            # in case of non ddp loss_explosion is true if the current rank has a loss explosion
            # only save dataloader state if loss_explosion is false, reason being rank 0 will only
            # save checkpoint if no rank has loss explosion
            if not loss_explosion:

                # if args.rank == 0:
                # NOTE(julieta) all ranks will save their models...
                logging.warning(f"rank {args.rank} save checkpoint to outpath {outpath}")
                torch.save(ae.state_dict(), "{}/aeparams.pt".format(outpath)) # outpath should be " run_base_dir/0/0"
                # torch.save(optim.state_dict(), "{}/optimparams.pt".format(outpath))
                torch.save(ae.state_dict(), "{}/aeparams_{:06d}.pt".format(outpath, iternum))
                # torch.save(optim.state_dict(), "{}/optimparams_{:06d}.pt".format(outpath, iternum))
                # checkpoint_job_config = read_job_config()
                # checkpoint_job_config["num_iterations"] = iternum
                # log_job_config(checkpoint_job_config)
            else:
                logging.warning(f"skipping checkpoint save, rank {args.rank} is seeing unstable loss value(s)")

        # if local_explosion:
        if loss_explosion:
            if enableddp:
                # to avoid any race conditions when multiple ranks are facing loss explosions
                # we do not want to load aeparams.pt while rank 0 is still writing the file
                dist.barrier()
            base_dir = os.environ["RSC_EXP_RUN_BASE_DIR"]
            if not base_dir:
                raise AssertionError("cannot reset without providing base_dir, check env var RSC_EXP_RUN_BASE_DIR")
            logging.warning(f"unstable loss function; resetting -- resume from the latest checkpoint : {outpath}/aeparams.pt")
            # reloading should called after backward is invoked to clean up CUDA memory of the failed iteration
            ae.load_state_dict(torch.load(f"{outpath}/aeparams.pt"), strict=False)
            _, optim = gen_optimizer(ae, optim_type, batchsize, args.rank, learning_rate, tensorboard_logger, encoder_lr=args.encoder_lr)
        # else:
        prevloss = loss.item()

        del(cudadata)


        if args.displayloss:
            logging.info("Rank {} Iteration {} loss = {:.4f}, ".format(args.rank, iternum, float(loss.item())) +
                    ", ".join(["{} = {:.4f}".format(k,
                        float(torch.sum(v[0]) / torch.sum(v[1].clamp(min=1e-6)) if isinstance(v, tuple) else torch.mean(v)))
                        for k, v in losses.items()]) \
                        + f" psnr: {psnr:.2f}" + f" time: {iter_total_time:.3f} s"
                        )

        # log losses to tensorboard even if we were not asked for profile stats
        if tensorboard_logger is not None and iternum % 50 == 0:
            tb_loss_stats = {
                "rank": args.rank,
                "loss": float(loss.item()),
            }
            for k, v in losses.items():
                if isinstance(v, tuple):
                    tb_loss_stats[k] = float(torch.sum(v[0]) / torch.sum(v[1].clamp(min=1e-6)))
                else:
                    tb_loss_stats[k] = torch.mean(v)

            for k, v in tb_loss_stats.items():
                tensorboard_logger.add_scalar("loss/" + k, v, iternum)

            if enableddp and args.worldsize > 1:
                averaged_losses = {k: torch.Tensor([v]).cuda() for k, v in tb_loss_stats.items()}
                for k, v in averaged_losses.items():
                    dist.all_reduce(v, op=dist.ReduceOp.AVG, group=allmembers)
                    averaged_losses[k] = v

                if rank == 0:
                    for k, v in averaged_losses.items():
                        tensorboard_logger.add_scalar(f"averaged_{k}/", v.item())


        if args.worldsize > 1 and iternum % 50 == 0:
            tb_loss_stats = {
                "rank": args.rank,
                "loss": float(loss.item()),
            }
            for k, v in losses.items():
                if isinstance(v, tuple):
                    tb_loss_stats[k] = float(torch.sum(v[0]) / torch.sum(v[1].clamp(min=1e-6)))
                else:
                    tb_loss_stats[k] = torch.mean(v)

            averaged_losses = {k: torch.Tensor([v]).cuda() for k, v in tb_loss_stats.items()}
            for k, v in averaged_losses.items():
                dist.all_reduce(v, op=dist.ReduceOp.AVG, group=allmembers)
                averaged_losses[k] = v

            if rank == 0:
                for k, v in averaged_losses.items():
                    logging.info(f"Iteration {iternum}, averaged {k}: {v.item():.3f}")


        if iternum and iternum % 2000 == 0:
            logging.info("Rank {} Iteration {} loss = {:.5f}, ".format(args.rank, iternum, float(loss.item())) +
                    ", ".join(["{} = {:.5f}".format(k,
                        float(torch.sum(v[0]) / torch.sum(v[1].clamp(min=1e-6)) if isinstance(v, tuple) else torch.mean(v)))
                        for k, v in losses.items()]))

            if perf_stats:
                log_perf_stats(iternum, args.rank, perf_stats)

            garbage_collection(iternum, args.rank)
            log_kerberos_ticket_details(iternum, args.rank)


        if iternum % 100 == 0:
            display_cudamem(f"TRAINING iteration {iternum} -- rank {args.rank}", cuda_device_count)


        iternum += 1

        if iternum >= maxiter:
            logging.info(f"Stopping training due to max iter limit, rank {args.rank} curr iter {iternum} max allowed iter {maxiter}")
            lend = time.time()
            totaltime = lend - lstart
            times = {"totaltime":totaltime, "maxiter":iternum}
            np.save(f"{outpath}/timesinfo_r{args.rank}", times, allow_pickle=True)

            logging.info("Rank {} Iteration {} loss = {:.5f}, ".format(args.rank, iternum, float(loss.item())) +
                    ", ".join(["{} = {:.5f}".format(k,
                        float(torch.sum(v[0]) / torch.sum(v[1].clamp(min=1e-6)) if isinstance(v, tuple) else torch.mean(v)))
                        for k, v in losses.items()]))

            if perf_stats:
                log_perf_stats(iternum, args.rank, perf_stats)
            break

    # cleanup
    writer.finalize()
