"""Train an autoencoder."""
import argparse
import gc
import importlib
import importlib.util
import os
import sys
import time
sys.dont_write_bytecode = True

import numpy as np
import torch

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)


#FOR AIRSTORE
#import torch.utils.data
from airstoredl import airdl

torch.backends.cudnn.benchmark = True # gotta go fast!

# Sanity check on PATH ENV VARIABLES
if os.getenv('RSC_AVATAR_PYUTILS_PATH', 'NOTFOUND') == 'NOTFOUND':
    assert(False, "RSC_AVATAR_PYUTILS_PATH NOT FOUND")

if os.getenv('RSC_JOB_UUID', 'NOTFOUND') == 'NOTFOUND':
    assert(False, "RSC_JOB_UUID NOT FOUND")

if os.getenv('RSC_AVATAR_METADATA_PATH', 'NOTFOUND') == 'NOTFOUND':
    assert(False, "RSC_AVATAR_AVATAR_METADATA_PATH NOT FOUND")

if os.getenv('RSC_AVATAR_RSCASSET_PATH', 'NOTFOUND') == 'NOTFOUND':
    assert(False, "RSC_AVATAR_AVATAR_RSCASSET_PATH NOT FOUND")

if os.getenv('RSC_AVATAR_READONLY_PATH', 'NOTFOUND') == 'NOTFOUND':
    assert(False, "RSC_AVATAR_READONLY_PATH NOT FOUND")

if os.getenv('RSC_AVATAR_DEBUGDATA_PATH', 'NOTFOUND') == 'NOTFOUND':
    assert(False, "RSC_AVATAR_DEBUGDATA_PATH NOT FOUND")

if os.getenv('RSC_AVATAR_EVAL_CONFIG_PATH', 'NOTFOUND') == 'NOTFOUND':
    assert(False, "RSC_AVATAR_EVAL_CONFIG_PATH NOT FOUND")

sys.path.append(os.getenv('RSC_AVATAR_PYUTILS_PATH'))

logging.info(f"sys path {sys.path}")

import pyutils

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


def save_meta(path, mds):

    print(" save meta : start ... ")

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

    print(" save meta : done ")


if __name__ == "__main__":
    # parse arguments
    print(" TRAIN PROC : OS PID  {}".format(os.getpid()))
    parser = argparse.ArgumentParser(description='Train an autoencoder')
    parser.add_argument('experconfig', type=str, help='experiment config file')
    parser.add_argument('--profile', type=str, default="Train", help='config profile')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--ablationcamerasinglegpu', action='store_true', help='running ablation camera experiments using 1 GPU')
    parser.add_argument('--noprogress', action='store_true', help='don\'t output training progress images')
    parser.add_argument('--nostab', action='store_true', help='don\'t check loss stability')
    parser.add_argument('--numworker', type=int, default=1, help='devices')
    parser.add_argument('--batchsize', type=int, default=6, help='batch size per GPU')
    parser.add_argument('--holdoutpath', type=str, default=None, help='directory to holdout info')
    parser.add_argument('--holdoutratio', type=str, default=None, help='cam hold out ratio')

    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()

    print(" args : holdoutpath : {}".format(args.holdoutpath))
    print(" args : holdoutratio : {}".format(args.holdoutratio))

    outpath = os.path.dirname(args.experconfig)
    py_file_handler = logging.FileHandler(f"{outpath}/py-log-r{args.rank}.txt")
    py_file_handler.setFormatter(formatter)
    root.addHandler(py_file_handler)

    log = pyutils.Logger2(tlogpath, args.resume)
    logging.info(f"Python {sys.version}")
    logging.info(f"PyTorch {torch.__version__}")
    logging.info(" ".join(sys.argv))
    logging.info(f"Output path: {outpath}")

    if args.ablationcamera or args.ablationcamerasinglegpu:
        assert(args.holdoutpath != None)
        assert(args.holdoutratio != None)
        logging.info("[ABLATION TEST CAMERA: holdoutpath {} holdoutratio {}".format(args.holdoutpath, args.holdoutratio))
    else:
        logging.info("[PLAIN TEST WITHOUT ABLATION]")

    print("Python", sys.version)
    print("PyTorch", torch.__version__)
    print(" ".join(sys.argv))
    print("Output path:", outpath)

    # load config
    starttime = time.time()
    experconfig = import_module(args.experconfig, "config")
    profile = getattr(experconfig, args.profile)(**{k: v for k, v in vars(args).items() if k not in parsed})
    if not args.noprogress:
        progressprof = experconfig.Progress()


    if args.batchsize:
        profile.batchsize = args.batchsize

    print("Config loaded ({:.2f} s)".format(time.time() - starttime))

    # build dataset & testing dataset
    starttime = time.time()

    # 2 identities : zev pac
    ings = ['codec_avatar_airstore_20211101_0822_zev994_avgtex_mugsy',
            'codec_avatar_airstore_20211102_0837_pac264_avgtex_mugsy']
    mds = np.load('/home/jinkyuk/backup-no-rsync/meta-data/dataset_meta_2id_pac_zev.npy', allow_pickle=True).tolist()

    # # ZEV only
    # ings = ['codec_avatar_airstore_20211101_0822_zev994_avgtex_mugsy']
    # mds = np.load('/home/jinkyuk/backup-no-rsync/meta-data/dataset_meta_1id_zev.npy', allow_pickle=True).tolist()

    # PAC only
    #ings = ['codec_avatar_airstore_20211102_0837_pac264_avgtex_mugsy']
    #mds = np.load('/home/jinkyuk/backup-no-rsync/meta-data/dataset_meta_1id_pac.npy', allow_pickle=True).tolist()

    #holdoutpath = '/home/jinkyuk/read-only/ablation-test'
    dataset = profile.get_dataset(mds, None, holdoutpath=args.holdoutpath, holdout_ratio=args.holdoutratio)

    numworker = args.numworker

    print("create data loader for airstore with {} io workers".format(numworker))

    # FOR MULTI IO WORKERS

    if hasattr(profile, "get_dataset_sampler"):
        #dataloader = torch.utils.data.DataLoader(dataset,
        dataloader = airdl.DataLoader(dataset,
                                      batch_size=profile.batchsize,
                                      sampler=profile.get_dataset_sampler(), drop_last=True,
                                      num_workers=numworker,
                                      gpu_rank=0,
                                      node_rank=0,
                                      gpu_per_node=8,
                                      ingest_list=ings)
    else:
        #dataloader = torch.utils.data.DataLoader(dataset,
        dataloader = airdl.DataLoader(dataset,
                                      batch_size=profile.batchsize, shuffle=True, drop_last=True,
                                      num_workers=numworker,
                                      gpu_rank=0,
                                      node_rank=0,
                                      gpu_per_node=8,
                                      ingest_list=ings)


    # data writer
    starttime = time.time()
    if not args.noprogress:
        writer = progressprof.get_writer()
        print("Writer instantiated ({:.2f} s)".format(time.time() - starttime))

    # build autoencoder
    starttime = time.time()
    ae = profile.get_autoencoder(dataset)
    # TODO: ddp
    #ae = torch.nn.DataParallel(ae, device_ids=args.devices).to("cuda").train()
    ae = ae.to("cuda").train()

    print("Autoencoder instantiated ({:.2f} s)".format(time.time() - starttime))

    # build optimizer
    starttime = time.time()
    #optim = profile.get_optimizer(ae.module)
    optim = profile.get_optimizer(ae)
    lossweights = profile.get_loss_weights()
    print("Optimizer instantiated ({:.2f} s)".format(time.time() - starttime))


    if args.resume:
        #ae.module.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
        print(" LOADING EXISTING PARAMS: {}/aeparams.pt and optimparams.pt ".format(outpath))
        ae.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
        optim.load_state_dict(torch.load("{}/optimparams.pt".format(outpath)))


    # train
    starttime = time.time()
    #evalpoints = np.geomspace(1., profile.maxiter, 100).astype(np.int32)

    evalpoints = list()
    for i in range(profile.maxiter//200):
        evalpoints.append(i*200)

    iternum = log.iternum
    prevloss = np.inf

    datalog = list()

    for epoch in range(10000):
        for data in dataloader:

            # forward
            frame_id = data['frameid']
            idindex = data['idindex']
            camindex = data['camindex'] #### NOT CAM NAME : it's camera index
            cams = data['cameraid']
            sids = data['sid']
            ee = {'frame_id':frame_id, 'idindex':idindex, 'camindex':camindex, 'cams':cams, 'sids':sids}
            datalog.append(ee)

            cudadata = tocuda(data)

            output, losses = ae(
                trainiter=iternum,
                outputlist=profile.get_outputlist() if hasattr(profile, "get_outputlist") else [],
                losslist=lossweights.keys(),
                **cudadata,
                **(profile.get_ae_args() if hasattr(profile, "get_ae_args") else {}))

            # compute final loss
            loss = sum([
                lossweights[k] * (torch.sum(v[0]) / torch.sum(v[1]).clamp(min=1e-6) if isinstance(v, tuple) else torch.mean(v))
                for k, v in losses.items()])

            # print current information
            print("Iteration {}: loss = {:.5f}, ".format(iternum, float(loss.item())) +
                    ", ".join(["{} = {:.5f}".format(k,
                        float(torch.sum(v[0]) / torch.sum(v[1].clamp(min=1e-6)) if isinstance(v, tuple) else torch.mean(v)))
                        for k, v in losses.items()]), end="")
            print("\n")

            if iternum % 100 == 0:
                endtime = time.time()
                ips = 100. / (endtime - starttime)
                print(", iter/sec = {:.2f}".format(ips))
                print(" elapsed tme for 100 iter: {}".format(endtime - starttime))
                starttime = time.time()

            # update parameters
            optim.zero_grad()
            loss.backward()

            #**************************
            params = [p for pg in optim.param_groups for p in pg["params"]]
            for p in params:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad.data[torch.isnan(p.grad.data)] = 0
                    p.grad.data[torch.isinf(p.grad.data)] = 0
            torch.nn.utils.clip_grad_norm_(params, 1)
            #**************************

            optim.step()

            #print progress
            if iternum % 100 == 0:
                writer.batch(iternum, iternum * profile.batchsize + torch.arange(0), f"progress_{iternum}.jpg", **cudadata, **output)

            # compute evaluation output
            if not args.noprogress and iternum in evalpoints:
                writer.batch(iternum, iternum * profile.batchsize + torch.arange(0), **cudadata, **output)

            #if not args.nostab and (loss.item() > 20 * prevloss or not np.isfinite(loss.item())):
            if not args.nostab and (loss.item() > 400 * prevloss or not np.isfinite(loss.item())):
                print("unstable loss function; resetting  at iternum : {}".format(iternum))

                #ae.module.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
                ae.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
                #optim = profile.get_optimizer(ae.module)
                optim = profile.get_optimizer(ae)

            prevloss = loss.item()

            # save intermediate results
            if iternum % 1000 == 0:
                #torch.save(ae.module.state_dict(), "{}/aeparams.pt".format(outpath))
                torch.save(ae.state_dict(), "{}/aeparams.pt".format(outpath))
                torch.save(optim.state_dict(), "{}/optimparams.pt".format(outpath))
                torch.save(ae.state_dict(), "{}/aeparams_{:06d}.pt".format(outpath, iternum))
                torch.save(optim.state_dict(), "{}/optimparams_{:06d}.pt".format(outpath, iternum))
                np.save("{}/datalog_{:06d}.npy".format(outpath, iternum), datalog, allow_pickle=True)
                datalog = list()

            if iternum >= profile.maxiter:
                break

            iternum += 1

        if iternum >= profile.maxiter:
            break

    # cleanup
    writer.finalize()
