"""
"""
# FOR CAM ABLATION EXPERIMENT:
# configuration change suggested by Jason June 20 sync meeting
#
# 1) set KL weight to 10^-3
# 2) remove VGG loss --> set its weight to zero or remove vgg network to save memory
#
# status : kl , vgg weight are set to zero
#

import os

#import data.genesis_multi as datamodel
# for using speicalized data loader for division exp (which requires old get_gen_dataset fn not dataset2)
import data.genesis_multi_division_ghsv2_airstore_dlrefactor as datamodel

import torch
import torch.nn as nn
import numpy as np


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

apath=os.getenv('RSC_AVATAR_RSCASSET_PATH')

holdoutcams = []
holdoutseg = []

# obsolete
capturepath = "/mnt/captures/studies/projects/GHS/Mugsy/m--20190529--1300--002421669--pilot--v1v2/"

gnewminisis_flag=True

_train_start_frac = 0
_train_end_frac = 1.0

_test_start_frac = 0.1
_test_end_frac = 1.0

_additional_keyfilters=[]

def get_dataset_base_params():
    #metapath = metabos.getenv('RSC_AVATAR_METADATA_PATH', "NOTFOUND")

    return dict(
        krtpath= None, # TODO: locate KRT in meta/meta/sid-mcd-mct/KRT   -- gen_ghsv2_paths  os.path.join(capturepath, "KRT_0.5x"),
        codecdir=None, #   -- os.path.join(capturepath, "processed_GHS_v2/codec/"),
        imagedir= None, # directly from airstore -- os.path.join(capturepath, "processed_GHS_v2/adhoc_0.5x/"),
        bgpath=None,         #"/mnt/captures/stephenlombardi/bg/002421669_20190529/0/{0}.png",
        returnbg=False,
        fixedcameras=["400263"],
        fixedcammean=100.,
        fixedcamstd=25.,
        maskbright=True,
        maskbrightbg=True,
        baseposepath=None) # TODO: locate first frame's pose info transform.txt   -- meta/meta/sid-mcd-mct/tracked_mesh/EXP_neut_peak -- gen_ghav2_paths os.path.join(capturepath, "processed_GHS_v2/codec/tracked_mesh/E001_Neutral_Eyes_Closed_Mouth_Open/013180_transform.txt"))

def get_dataset(ids, holdoutpath=None, holdout_ratio=None,
        num_workers=None,  # for AIRSTORE
        gpu_rank=None, # for AIRSTORE
        enable_deterministic=False, # FOR AIRSTORE
        gpu_world_size=None, # FOR AIRSTORE
        disable_shuffle_air=False, # FOR AIRSTORE
        ingest_list = None, # for AIRSTORE
        camerafilter =lambda x: x not in holdoutcams,
        segmentfilter=lambda x: x not in holdoutseg,
        keyfilter=[],
        maxframes=-1,
        relativecamera=True,
        subsampletype=None,
        downsample=2,
        start_frac = 0,
        end_frac = 1.0,
        **kwargs):

    assert(len(keyfilter) == 0) # not verified yet except default ones: bg, camera, modelmatrix, pixelcoords, iamge, avgtex, verts, neut_avgtex, neut_verts

    config = get_dataset_base_params()
    config['camerafilter'] = camerafilter
    config['segmentfilter'] = segmentfilter
    #print(" get_dataset : keyfilter {}".format(keyfilter))
    config['keyfilter'] = ["bg", "camera", "modelmatrix", "pixelcoords", "image", "avgtex", "verts", "neut_avgtex", "neut_verts"] + keyfilter
    #print(" get_dataset : config[keyfilter] {}".format(config['keyfilter']))
    config['maxframes'] = maxframes
    config['relativecamera'] = relativecamera
    config['subsampletype'] = subsampletype
    config['subsamplesize'] = 384
    config['downsample'] = downsample

    # airstore creation
    config['num_workers'] = num_workers
    config['gpu_rank'] = gpu_rank
    config['enable_deterministic'] = enable_deterministic
    config['gpu_world_size'] = gpu_world_size
    config['disable_shuffle_air'] = disable_shuffle_air
    config['ingest_list'] = ingest_list

    # move holdout path and ratio
    config['holdoutpath'] = holdoutpath
    config['holdout_ratio'] = holdout_ratio

    dataset = datamodel.gen_genesis_dataset(ids, config)
    return dataset

def get_renderoptions():
    return dict(
            dt=1.)


def create_uv_baridx(geofile, trifile, barfiles):

    import pyutils
    import cv2

    _, vt, vi, vti = pyutils.load_obj(geofile)

    vt = np.array(vt, dtype=np.float32)
    vi = np.array(vi, dtype=np.int32)
    vti = np.array(vti, dtype=np.int32)

    vt[:,1] = 1 - vt[:,1] #note: flip y-axis
    uvtri = np.genfromtxt(trifile, dtype=np.int32)
    bar = []
    for i in range(3):
        bar.append(np.genfromtxt(barfiles[i], dtype=np.float32))

    idx0 = cv2.flip(vi[uvtri,0],flipCode=0)
    idx1 = cv2.flip(vi[uvtri,1],flipCode=0)
    idx2 = cv2.flip(vi[uvtri,2],flipCode=0)
    bar0 = cv2.flip(bar[0],flipCode=0)
    bar1 = cv2.flip(bar[1],flipCode=0)
    bar2 = cv2.flip(bar[2],flipCode=0)

    return {'uv_idx': np.concatenate((idx0[None,:,:],
                                      idx1[None,:,:],
                                      idx2[None,:,:]), axis=0),
            'uv_bary': np.concatenate((bar0[None,:,:],
                                       bar1[None,:,:],
                                       bar2[None,:,:]), axis=0),
            'uv_coord': vt, 'uv_tri': vti, 'tri': vi }

def get_autoencoder(dataset):
    import torch
    import torch.nn as nn
    #import models.volumetric_multi2 as aemodel
    import models.volumetric_multi3 as aemodel # clean one no profiling code
    import models.encoders.geotex1_multi as encoderlib
    import models.decoders.bm3_multi as decoderlib
    import models.raymarchers.mvpraymarcher_new as raymarcherlib
    import models.colorcals.colorcal_multi as colorcalib
    import models.bg.mlp2d_multi as bglib
    import pyutils

    allcameras = dataset.get_allcameras()
    ncams = len(allcameras)
    width, height = dataset.get_img_size()

    print("@@@ Get autoencoder ABLATION CONFIG FILE : lenth of data set : {}".format(len(dataset.identities)))

    colorcal = colorcalib.Colorcal2(len(dataset.get_allcameras()), len(dataset.identities))
    apath=os.getenv('RSC_AVATAR_RSCASSET_PATH')
    objpath = f"{apath}/geotextop.obj"

    v, vt, vi, vti = pyutils.load_obj(objpath)
    vt = np.array(vt, dtype=np.float32)
    vi = np.array(vi, dtype=np.int32)
    vti = np.array(vti, dtype=np.int32)

    vertmean = torch.from_numpy(dataset.vertmean)
    vertstd = dataset.vertstd

    #load per-textel triangulation indices
    resolution = 1024
    geofile = f"{apath}/retop.obj"
    uvpath = f"{apath}/fd-data/"

    trifile = f'{uvpath}/uv_tri_{resolution}_orig.txt'
    barfiles = []
    for i in range(3):
        barfiles.append(f'{uvpath}/uv_bary{i}_{resolution}_orig.txt')
    uvdata = create_uv_baridx(geofile, trifile, barfiles)

    id_encoder = encoderlib.EncoderIdentity2(uvdata['uv_idx'], uvdata['uv_bary'], wsize=128)
    encoder = encoderlib.EncoderExpression(uvdata['uv_idx'], uvdata['uv_bary'])
    volradius = 256.

    #decoder = decoderlib.Decoder(vt, vertmean, vertstd, volradius=volradius,
    #                             nprims=128*128, primsize=(8, 8, 8), motiontype="deconv",
    #                             postrainstart=100, warp=None)

    # decoder = decoderlib.Decoder4(vt, vi, vti, vertmean, vertstd, volradius=volradius,
    #                               nprims=16*16, primsize=(32, 32, 32), motiontype="deconv",
    #                               postrainstart=1000, warp=None)

    # decoder = decoderlib.Decoder4(vt, vi, vti, vertmean, vertstd, volradius=volradius,
    #                              nprims=128*128, primsize=(8,8,8), motiontype="deconv",
    #                              postrainstart=200, warp=None)
    # decoder = decoderlib.Decoder5(vt, vi, vti, vertmean, vertstd, volradius=volradius,
    #                               nprims=16*16, primsize=(32,32,32), motiontype="deconv",
    #                               postrainstart=100, warp=None)
    decoder = decoderlib.Decoder5(vt, vi, vti, vertmean, vertstd, volradius=volradius,
                                  nprims=128*128, primsize=(8,8,8), motiontype="deconv",
                                  postrainstart=100, warp=None)

    # decoder = decoderlib.Decoder3(vt, vi, vti, vertmean, vertstd, volradius=volradius,
    #                               nprims=16*16, primsize=(32, 32, 32), motiontype="deconv",
    #                               postrainstart=1000, warp=None)

    #volsampler = volsamplerlib.VolSampler()
    #raymarcher = raymarcherlib.Raymarcher(volradius=volradius)

    config = ObjDict({'render': {'raymarcher_options': {"volume_radius": volradius, 'chlast': False}}}) ################## change to true for fast rendering?
    raymarcher = raymarcherlib.Raymarcher(config)

    bgmodel = bglib.BackgroundModelSimple(len(allcameras), len(dataset.identities))
    #bgmodel = bglib.BGModel(len(dataset.identities), width, height, allcameras, trainstart=0, bgdict=False)
    ae = aemodel.Autoencoder(
        dataset,
        id_encoder,
        encoder,
        decoder,
        raymarcher,
        colorcal,
        bgmodel,
        #encoderinputs=["fixedcamimage"],
        encoderinputs=["verts", "avgtex", "neut_verts", "neut_avgtex"],
        topology={"vt": vt, "vi": vi, "vti": vti},
        imagemean=100.,
        imagestd=25.,
        use_vgg=False)

    # DO NOT RUN VGG AT ALL and remove vgg in loss_weight for ABLATION TEST : @@@@
    print("id_encoder params:", sum(p.numel() for p in ae.id_encoder.parameters() if p.requires_grad))
    print("encoder params:", sum(p.numel() for p in ae.encoder.parameters() if p.requires_grad))
    print("decoder params:", sum(p.numel() for p in ae.decoder.parameters() if p.requires_grad))
    print("colorcal params:", sum(p.numel() for p in ae.colorcal.parameters() if p.requires_grad))
    print("bgmodel params:", sum(p.numel() for p in ae.bgmodel.parameters() if p.requires_grad))
    print("total params:", sum(p.numel() for p in ae.parameters() if p.requires_grad))

    return ae

# profiles
class Train():
    batchsize=6
    maxiter=50000000
    #num_worker=2

    def get_ings(self, ids2use=-1, idfilepath=None):
        assert idfilepath != None
        idlist=list()

        if idfilepath != None:
            ids=open(f'{idfilepath}', 'rt').readlines()
            for e in ids:
                idlist.append(e.rstrip())

        assert ids2use <= len(idlist)

        tab={"tablelist":idlist}
        if ids2use != -1:
            assert ids2use <= len(tab['tablelist'])
            tablist=tab['tablelist'][:ids2use]
            tab['tablelist'] = tablist

        print("@@@@@@@@@@@@ DEBUGGING: idlist from gen_ings: {}".format(idlist))
        return tab

    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_outputlist(self): return ["irgbrec", "irgbsqerr", "sampledimg", "bg"]
    def get_ae_args(self): return dict(renderoptions=get_renderoptions())
    def get_dataset(self, ids, holdoutpath=None, holdout_ratio=None,
                    num_workers=None,
                    gpu_rank=None,
                    enable_deterministic=False,
                    gpu_world_size=None,
                    disable_shuffle_air=False, 
                    ids2use=None, 
                    idfilepath=None):

        mdata = self.get_ings(ids2use=ids2use, idfilepath=idfilepath)
        inglist = mdata['tablelist']

        print("@@@@@@@@@@@@@ CONFIG: IDS2USE: {}  -- length of inglist : {}  length of ids {} -- inglist: {},   IDSlist: {}".format(ids2use, len(inglist), len(ids), inglist, ids))
        assert len(ids) == len(inglist)

        return get_dataset(ids, holdoutpath=holdoutpath, holdout_ratio=holdout_ratio,
                    num_workers=num_workers, gpu_rank=gpu_rank,
                    enable_deterministic=enable_deterministic,
                    gpu_world_size=gpu_world_size,
                    disable_shuffle_air=disable_shuffle_air,
                    ingest_list = inglist
                    )

    def get_optimizer(self, ae):
        import itertools
        import torch.optim
        lr = 0.001
        aeparams = itertools.chain(
            [{"params": x} for k, x in ae.id_encoder.named_parameters()],
            [{"params": x} for k, x in ae.encoder.named_parameters()],
            [{"params": x} for k, x in ae.decoder.named_parameters()],
            [{"params": x} for x in ae.bgmodel.parameters()],
            [{"params": x} for x in ae.colorcal.parameters()])
        return torch.optim.Adam(aeparams, lr=lr, betas=(0.9, 0.999))
    def get_loss_weights(self):
        #return {"irgbl1": 1.0, "vertl1": 0.1, "kldiv": 0.0001, "primvolsum": 0.01, "vgg": 1.0}
        # for ablation: kl loss weight: 10^-3
        #return {"irgbl1": 1.0, "vertl1": 0.1, "kldiv": 0.001, "primvolsum": 0.01, "vgg": 0.0}
        return {"irgbl1": 1.0, "vertl1": 0.1, "kldiv": 0.001, "primvolsum": 0.01}

class ProgressWriterNoBG():
    def batch(self, iternum, itemnum, imagename = None, **kwargs):
        import numpy as np
        from PIL import Image
        rows = []
        row = []
        for i in range(kwargs["image"].size(0)):
            if ("irgbsqerr" in kwargs) and ("sampledimg" in kwargs):
                row.append(
                    np.concatenate((
                        (torch.sqrt(kwargs["irgbsqerr"][i].data + 1e-3).to("cpu").numpy().transpose((1, 2, 0)) * 100.).clip(0,255),
                        kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0)),
                        kwargs["sampledimg"][i].data.to("cpu").numpy().transpose((1, 2, 0))), axis=1))
            else:
                row.append(
                    np.concatenate((
                        kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                        kwargs["image"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]), axis=1))
            if len(row) == 6:
                rows.append(np.concatenate(row, axis=1))
                row = []
        imgout = np.concatenate(rows, axis=0) * 2.
        outpath = os.path.dirname(__file__)
        if imagename is None:
            Image.fromarray(np.clip(imgout, 0, 255).astype(np.uint8)).save(os.path.join(outpath, "prog_{:06}.jpg".format(iternum)))
        else:
            Image.fromarray(np.clip(imgout, 0, 255).astype(np.uint8)).save(os.path.join(outpath, imagename))
    def finalize(self):
        pass


class ProgressWriter():
    def batch(self, iternum, itemnum, imagename = None, **kwargs):
        import numpy as np
        from PIL import Image
        rows = []
        row = []

        #print(" KWARGS IMAGE SIZE (0) : {}".format(kwargs["image"].size(0)))

        for i in range(kwargs["image"].size(0)):
            if ("irgbsqerr" in kwargs) and ("sampledimg" in kwargs):  # training output
                if "bg" in kwargs and kwargs['bg'] != None:
                    row.append(
                        np.concatenate((
                            (torch.sqrt(kwargs["irgbsqerr"][i].data + 1e-3).to("cpu").numpy().transpose((1, 2, 0)) * 100.).clip(0,255),
                            kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0)),
                            kwargs["bg"][i].data.to("cpu").numpy().transpose((1, 2, 0)),
                            kwargs["sampledimg"][i].data.to("cpu").numpy().transpose((1, 2, 0))), axis=1))
                else:
                    row.append(
                        np.concatenate((
                            (torch.sqrt(kwargs["irgbsqerr"][i].data + 1e-3).to("cpu").numpy().transpose((1, 2, 0)) * 100.).clip(0,255),
                            kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0)),
                            kwargs["sampledimg"][i].data.to("cpu").numpy().transpose((1, 2, 0))), axis=1))
            else:  # testoutputs
                if "bg" in kwargs and kwargs['bg'] != None: 
                    row.append(
                        np.concatenate((
                            kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                            kwargs["bg"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                            kwargs["image"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]), axis=1))
                else:
                    row.append(
                        np.concatenate((
                            kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                            kwargs["image"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]), axis=1))                    

            if len(row) == 4:
                rows.append(np.concatenate(row, axis=1))
                row = []
        imgout = np.concatenate(rows, axis=0) * 2.
        outpath = os.path.dirname(__file__)
        if imagename is None:
            Image.fromarray(np.clip(imgout, 0, 255).astype(np.uint8)).save(os.path.join(outpath, "prog_{:06}.jpg".format(iternum)))
        else:
            Image.fromarray(np.clip(imgout, 0, 255).astype(np.uint8)).save(os.path.join(outpath, imagename))
    def finalize(self):
        pass



# TODO: option to control frequency of output
class Progress():
    batchsize=4
    def get_outputlist(self): return ["irgbrec", "bg"]
    def get_ae_args(self): return dict(renderoptions=get_renderoptions())
    #def get_dataset(self): return get_dataset(start_frac = 0, end_frac = 1.0)
    def get_dataset(self, mds, ids): return get_dataset(mds, ids)
    def get_writer(self): return ProgressWriter()

class Render():
    def __init__(self,
                 input_ididx = 1,
                 output_ididx = [1,0,-1,-2,-3], # [12,13,14,15,16,17,18,19], #[4, 5, 6, 7, 8, 9, 10, 11], #
                 test_output_ididx = [0,1,-1,-2,-3], #[-5, -6, -7, -8, -9, -10, -11, -12], #
                 #outfilesuffix=None,
                 cam=None, camdist=768., camperiod=128, camrevs=0.5,#0.25,
                 segments=['ROM07_Facial_Expressions'],
                 #segments=['S9_They_had_slapped_their_thighs._'],#segments=["S01_She_always_jokes_about_too_much_garlic_in_his_food"],
                 maxframes=-1, relativecamera=True,
                 drawmesh=False, viewtemplate=False, colorprims=False,
                 keyfilter=[]):
        #self.outfilename = outfilename
        #self.outfilesuffix = outfilesuffix

        self.input_ididx = input_ididx
        self.output_ididx = output_ididx
        self.test_output_ididx = test_output_ididx

        self.vidfile = os.path.dirname(__file__) + '/render.mp4'
        self.imgdir = os.path.dirname(__file__) + '/render_images'

        self.cam = cam
        self.camdist = camdist
        self.camperiod = camperiod
        self.camrevs = camrevs
        self.segments = segments
        self.segmentfilter = lambda x: True if len(segments) == 0 else x in segments
        self.maxframes = maxframes
        self.relativecamera = relativecamera
        self.drawmesh = drawmesh
        self.viewtemplate = viewtemplate
        self.colorprims = colorprims
        self.keyfilter = keyfilter
        self.segments = segments
        # e.g., ["irgbarec_nomesh", "irgbarec_novol", "irgbrec", "irgbsqerr", "image"]

    def get_identities(self): return self.input_ididx, self.output_ididx, self.test_output_ididx
    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_outputlist(self): return ["irgbrec"]
    #def get_outputlist(self): return [k[0] if isinstance(k, tuple) else k for k in self.keyfilter]
    def get_ae_args(self): return dict(renderoptions={**get_renderoptions(), **{"colorprims": self.colorprims}})

    #def get_test_dataset(self): return get_dataset(start_frac = _end_frac+1e-3, end_frac = 1.0)
    def get_test_dataset(self): return get_dataset(start_frac = _test_start_frac, _test_end_frac = 1.0)

    def get_dataset(self, ididx):
        import data.utils
        import eval.cameras.rotate as cameralib
        if self.cam == "all":
            camerafilter = lambda x: x.startswith("40")
        else:
            camerafilter = lambda x: x == self.cam

        #elif self.cam is None:
        #    camerafilter = lambda x: x == "400029"

        dataset = get_dataset(camerafilter=camerafilter,
                              segmentfilter=self.segmentfilter,
                              keyfilter=[k[0] if isinstance(k, tuple) else k for k in self.keyfilter],
                              maxframes=self.maxframes,
                              relativecamera=self.relativecamera,
                              start_frac = 0, end_frac = _end_frac).datasets[ididx]
        if self.cam is None:
            camdataset = cameralib.Dataset(len(dataset), camdist=self.camdist, period=self.camperiod, revs=self.camrevs)
            return data.utils.ColCatDataset(camdataset, dataset)
        else:
            return dataset
    # def get_writer(self):
    #     import eval.writers.videowriter as writerlib
    #     if self.outfilename is None:
    #         outfilename = (
    #                 "render_{}_{}".format("-".join([x[:4].replace('_', '') for x in self.segments]), self.cam) +
    #                 (self.outfilesuffix if self.outfilesuffix is not None else "") +
    #                 ".mp4")
    #     else:
    #         outfilename = self.outfilename

    #     return writerlib.Writer(
    #         os.path.join(os.path.dirname(__file__), outfilename),
    #         keyfilter=self.keyfilter)



#render grouped collection of identities
class Render2():
    def __init__(self,
                 input_ididx = 1,
                 idtype = 'single', #'accesories', #'hair', #'facial_hair', #'dark_skintone',
                 cam=None, camdist=768., camperiod=128, camrevs=0.25,
                 segments=['ROM07_Facial_Expressions'],
                 #segments=['S9_They_had_slapped_their_thighs._'],#segments=["S01_She_always_jokes_about_too_much_garlic_in_his_food"],
                 maxframes=-1, relativecamera=True,
                 drawmesh=False, viewtemplate=False, colorprims=False,
                 keyfilter=[]):
        #self.outfilename = outfilename
        #self.outfilesuffix = outfilesuffix

        self.input_ididx = input_ididx


        if idtype == 'single':
            self.output_ididx = [1]
        elif idtype == 'dark_skintone':
            self.output_ididx = [ 1,    15,  54,  58,  67,  72,  79, 112, 132, 135, 138, 139, 144, 150, 152, 162, 168, 189, 194, 209, 222, 227, 229, 235, 243]
        elif idtype == 'facial_hair':
            self.output_ididx = [48, 68, 71, 79, 93, 102, 103, 108, 112, 122, 132, 137, 158, 180, 227, 237, 246]
        elif idtype == 'hair':
            self.output_ididx = [3, 4, 5, 13, 35, 60, 73, 76, 84, 88, 104, 119, 124, 126, 129, 131, 191, 207, 214, 218, 229, 230, 232, 243]
        elif idtype == 'accesories':
            self.output_ididx = [18, 72, 77, 98, 156, 176, 189, 232, 242]
        else:
            print('Unknown idtype')

        self.vidfile = os.path.dirname(__file__) + f'/render_{idtype}.mp4'
        self.imgdir = os.path.dirname(__file__) + f'/render_{idtype}_images'

        self.cam = cam
        self.camdist = camdist
        self.camperiod = camperiod
        self.camrevs = camrevs
        self.segments = segments
        self.segmentfilter = lambda x: True if len(segments) == 0 else x in segments
        self.maxframes = maxframes
        self.relativecamera = relativecamera
        self.drawmesh = drawmesh
        self.viewtemplate = viewtemplate
        self.colorprims = colorprims
        self.keyfilter = keyfilter
        self.segments = segments
        # e.g., ["irgbarec_nomesh", "irgbarec_novol", "irgbrec", "irgbsqerr", "image"]

    def get_identities(self): return self.input_ididx, self.output_ididx
    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_outputlist(self): return ["irgbrec"]
    #def get_outputlist(self): return [k[0] if isinstance(k, tuple) else k for k in self.keyfilter]
    def get_ae_args(self): return dict(renderoptions={**get_renderoptions(), **{"colorprims": self.colorprims}})
    def get_test_dataset(self): return get_dataset(start_frac = _end_frac+1e-3, end_frac = 1.0)
    def get_dataset(self, ididx):
        import data.utils
        import eval.cameras.rotate as cameralib
        if self.cam == "all":
            camerafilter = lambda x: x.startswith("40")
        else:
            camerafilter = lambda x: x == self.cam

        #elif self.cam is None:
        #    camerafilter = lambda x: x == "400029"

        dataset = get_dataset(camerafilter=camerafilter,
                              segmentfilter=self.segmentfilter,
                              keyfilter=[k[0] if isinstance(k, tuple) else k for k in self.keyfilter],
                              maxframes=self.maxframes,
                              relativecamera=self.relativecamera,
                              start_frac = 0, end_frac = _end_frac).datasets[ididx]
        if self.cam is None:
            camdataset = cameralib.Dataset(len(dataset), camdist=self.camdist, period=self.camperiod, revs=self.camrevs)
            return data.utils.ColCatDataset(camdataset, dataset)
        else:
            return dataset
    # def get_writer(self):
    #     import eval.writers.videowriter as writerlib
    #     if self.outfilename is None:
    #         outfilename = (
    #                 "render_{}_{}".format("-".join([x[:4].replace('_', '') for x in self.segments]), self.cam) +
    #                 (self.outfilesuffix if self.outfilesuffix is not None else "") +
    #                 ".mp4")
    #     else:
    #         outfilename = self.outfilename

    #     return writerlib.Writer(
    #         os.path.join(os.path.dirname(__file__), outfilename),
    #         keyfilter=self.keyfilter)


#only render test subjects
class Render3():
    def __init__(self,
                 input_ididx = 1,
                 cam=None, camdist=768., camperiod=128, camrevs=0.25,
                 segments=['ROM07_Facial_Expressions'],
                 #segments=['S9_They_had_slapped_their_thighs._'],#segments=["S01_She_always_jokes_about_too_much_garlic_in_his_food"],
                 maxframes=-1, relativecamera=True,
                 drawmesh=False, viewtemplate=False, colorprims=False,
                 keyfilter=[]):
        #self.outfilename = outfilename
        #self.outfilesuffix = outfilesuffix

        self.input_ididx = input_ididx

        self.vidfile = os.path.dirname(__file__) + f'/render_unseen.mp4'
        self.imgdir = os.path.dirname(__file__) + f'/render_unseen_images'

        self.cam = cam
        self.camdist = camdist
        self.camperiod = camperiod
        self.camrevs = camrevs
        self.segments = segments
        self.segmentfilter = lambda x: True if len(segments) == 0 else x in segments
        self.maxframes = maxframes
        self.relativecamera = relativecamera
        self.drawmesh = drawmesh
        self.viewtemplate = viewtemplate
        self.colorprims = colorprims
        self.keyfilter = keyfilter
        self.segments = segments
        # e.g., ["irgbarec_nomesh", "irgbarec_novol", "irgbrec", "irgbsqerr", "image"]

    def get_identities(self): return self.input_ididx
    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_outputlist(self): return ["irgbrec"]
    #def get_outputlist(self): return [k[0] if isinstance(k, tuple) else k for k in self.keyfilter]
    def get_ae_args(self): return dict(renderoptions={**get_renderoptions(), **{"colorprims": self.colorprims}})
    def get_test_dataset(self): return get_dataset(start_frac = _end_frac+1e-3, end_frac = 1.0)
    def get_dataset(self, ididx):
        import data.utils
        import eval.cameras.rotate as cameralib
        if self.cam == "all":
            camerafilter = lambda x: x.startswith("40")
        else:
            camerafilter = lambda x: x == self.cam

        #elif self.cam is None:
        #    camerafilter = lambda x: x == "400029"

        dataset = get_dataset(camerafilter=camerafilter,
                              segmentfilter=self.segmentfilter,
                              keyfilter=[k[0] if isinstance(k, tuple) else k for k in self.keyfilter],
                              maxframes=self.maxframes,
                              relativecamera=self.relativecamera,
                              start_frac = 0, end_frac = _end_frac).datasets[ididx]
        if self.cam is None:
            camdataset = cameralib.Dataset(len(dataset), camdist=self.camdist, period=self.camperiod, revs=self.camrevs)
            return data.utils.ColCatDataset(camdataset, dataset)
        else:
            return dataset
    # def get_writer(self):
    #     import eval.writers.videowriter as writerlib
    #     if self.outfilename is None:
    #         outfilename = (
    #                 "render_{}_{}".format("-".join([x[:4].replace('_', '') for x in self.segments]), self.cam) +
    #                 (self.outfilesuffix if self.outfilesuffix is not None else "") +
    #                 ".mp4")
    #     else:
    #         outfilename = self.outfilename

    #     return writerlib.Writer(
    #         os.path.join(os.path.dirname(__file__), outfilename),
    #         keyfilter=self.keyfilter)



#only render wild subjects
class Render4():
    def __init__(self,
                 input_ididx = 1,
                 eval_ididx = [0,1,2,3,4],
                 cam=None, camdist=768., camperiod=512, camrevs=1.0,
                 segments=['ROM07_Facial_Expressions'],
                 #segments=['S9_They_had_slapped_their_thighs._'],#segments=["S01_She_always_jokes_about_too_much_garlic_in_his_food"],
                 maxframes=-1, relativecamera=True,
                 drawmesh=False, viewtemplate=False, colorprims=False,
                 keyfilter=[]):
        #self.outfilename = outfilename
        #self.outfilesuffix = outfilesuffix

        self.input_ididx = input_ididx

        self.vidfile = os.path.dirname(__file__) + f'/render_wild.mp4'
        self.imgdir = os.path.dirname(__file__) + f'/render_wild_images'

        self.cam = cam
        self.camdist = camdist
        self.camperiod = camperiod
        self.camrevs = camrevs
        self.segments = segments
        self.segmentfilter = lambda x: True if len(segments) == 0 else x in segments
        self.maxframes = maxframes
        self.relativecamera = relativecamera
        self.drawmesh = drawmesh
        self.viewtemplate = viewtemplate
        self.colorprims = colorprims
        self.keyfilter = keyfilter
        self.segments = segments
        # e.g., ["irgbarec_nomesh", "irgbarec_novol", "irgbrec", "irgbsqerr", "image"]


        self.eval_geofile = ["/mnt/home/jsaragih/code/codec_multi/train/YaserNeutralGeo2.obj",
                             '/mnt/home/chen/CARE/runs/avatar_wild_phone_old_4/avatar_wild_geo_refine/002643814_20210319/mesh.obj',
                             "/mnt/home/jsaragih/code/codec_multi/train/DaniNeutralGeo.obj",
                             "/mnt/home/chen/CARE/runs/avatar_wild_phone_shugao/avatar_wild_geo_refine/002643814_20210319/mesh.obj",
                             "/mnt/home/chen/CARE/runs/avatar_wild_phone_xiaomin/avatar_wild_geo_refine/002643814_20210319/mesh.obj"]
        self.eval_texfile = ["/mnt/home/jsaragih/code/codec_multi/train/YaserNeutralTex2.png",
                             '/mnt/home/chen/CARE/runs/avatar_wild_phone_old_4/avatar_wild_geo_refine/002643814_20210319/tex.png',
                             "/mnt/home/jsaragih/code/codec_multi/train/DaniNeutralTex.png",
                             "/mnt/home/chen/CARE/runs/avatar_wild_phone_shugao/avatar_wild_geo_refine/002643814_20210319/tex.png",
                             "/mnt/home/chen/CARE/runs/avatar_wild_phone_xiaomin/avatar_wild_geo_refine/002643814_20210319/tex.png"]
        self.eval_ididx = eval_ididx

    def get_identities(self): return self.input_ididx
    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_outputlist(self): return ["irgbrec"]
    #def get_outputlist(self): return [k[0] if isinstance(k, tuple) else k for k in self.keyfilter]
    def get_ae_args(self): return dict(renderoptions={**get_renderoptions(), **{"colorprims": self.colorprims}})
    def get_test_dataset(self): return get_dataset(start_frac = _end_frac+1e-3, end_frac = 1.0)
    def get_dataset(self, ididx):
        import data.utils
        import eval.cameras.rotate as cameralib
        if self.cam == "all":
            camerafilter = lambda x: x.startswith("40")
        else:
            camerafilter = lambda x: x == self.cam

        #elif self.cam is None:
        #    camerafilter = lambda x: x == "400029"

        dataset = get_dataset(camerafilter=camerafilter,
                              segmentfilter=self.segmentfilter,
                              keyfilter=[k[0] if isinstance(k, tuple) else k for k in self.keyfilter],
                              maxframes=self.maxframes,
                              relativecamera=self.relativecamera,
                              start_frac = 0, end_frac = _end_frac).datasets[ididx]
        if self.cam is None:
            camdataset = cameralib.Dataset(len(dataset), camdist=self.camdist, period=self.camperiod, revs=self.camrevs)
            return data.utils.ColCatDataset(camdataset, dataset)
        else:
            return dataset
    # def get_writer(self):
    #     import eval.writers.videowriter as writerlib
    #     if self.outfilename is None:
    #         outfilename = (
    #                 "render_{}_{}".format("-".join([x[:4].replace('_', '') for x in self.segments]), self.cam) +
    #                 (self.outfilesuffix if self.outfilesuffix is not None else "") +
    #                 ".mp4")
    #     else:
    #         outfilename = self.outfilename

    #     return writerlib.Writer(
    #         os.path.join(os.path.dirname(__file__), outfilename),
    #         keyfilter=self.keyfilter)





class RenderCam():
    def __init__(self,
                 ididx = 1,
                 id_type = 'all',
                 cam=["400029", "400050"],
                 camdist=768., camperiod=128, camrevs=0.25,
                 segment_type='rom', #'rom', 'expressions', 'gaze', 'tongue', 'speech' #segments=['E0'],#['ROM'],
                 maxframes=-1, relativecamera=True,
                 drawmesh=False, viewtemplate=False, colorprims=False,
                 keyfilter=[]):
        #self.outfilename = outfilename
        #self.outfilesuffix = outfilesuffix

        self.set_segment(segment_type)
        self.set_identities(id_type, ididx)

        self.cam = cam
        self.camdist = camdist
        self.camperiod = camperiod
        self.camrevs = camrevs


        self.maxframes = maxframes
        self.relativecamera = relativecamera
        self.drawmesh = drawmesh
        self.viewtemplate = viewtemplate
        self.colorprims = colorprims
        self.keyfilter = keyfilter
        self.batchsize = 8

        self.label = False



        self.eval_geofile = ["/mnt/home/jsaragih/code/codec_multi/train/YaserNeutralGeo2.obj",
                             '/mnt/home/chen/CARE/runs/avatar_wild_phone_old_4/avatar_wild_geo_refine/002643814_20210319/mesh.obj',
                             "/mnt/home/jsaragih/code/codec_multi/train/DaniNeutralGeo.obj",
                             "/mnt/home/chen/CARE/runs/avatar_wild_phone_shugao/avatar_wild_geo_refine/002643814_20210319/mesh.obj",
                             "/mnt/home/chen/CARE/runs/avatar_wild_phone_xiaomin/avatar_wild_geo_refine/002643814_20210319/mesh.obj"]
        self.eval_texfile = ["/mnt/home/jsaragih/code/codec_multi/train/YaserNeutralTex2.png",
                             '/mnt/home/chen/CARE/runs/avatar_wild_phone_old_4/avatar_wild_geo_refine/002643814_20210319/tex.png',
                             "/mnt/home/jsaragih/code/codec_multi/train/DaniNeutralTex.png",
                             "/mnt/home/chen/CARE/runs/avatar_wild_phone_shugao/avatar_wild_geo_refine/002643814_20210319/tex.png",
                             "/mnt/home/chen/CARE/runs/avatar_wild_phone_xiaomin/avatar_wild_geo_refine/002643814_20210319/tex.png"]


    def label_images(self): return self.label

    def get_vidfile(self): return os.path.dirname(__file__) + f'/render_{self.segment_type}_{self.id_type}_{self.ididx}.mp4'
    def get_imgdir(self):  return os.path.dirname(__file__) + f'/render_{self.segment_type}_{self.id_type}_{self.ididx}'

    def set_identities(self, id_type, ididx):
        self.id_type = id_type
        self.ididx = ididx
        self.eval_ididx = []
        if id_type == 'recon':
            self.train_ididx = []
            self.test_ididx = []
        elif id_type == 'train':
            self.train_ididx = [88, 71]
            self.test_ididx = []
        elif id_type == 'test':
            self.train_ididx = []
            self.test_ididx = [7, 6]
        elif id_type == 'eval':
            self.train_ididx = []
            self.test_ididx = []
            self.eval_ididx = [i for i in range(len(self.eval_geofile))]
        elif id_type == 'all':
            self.train_ididx = [88, 71]
            self.test_ididx = [7, 6]
            self.eval_ididx = [0, 1]
        else:
            print(f'Unsupported id_type: {id_type}')
            quit()


    def set_segment(self, segment_type):
        self.segment_type = segment_type
        if segment_type == 'rom':
            self.segments = ['ROM']
        elif segment_type == 'expressions':
            self.segments = ['E0']
        elif segment_type == 'speech':
            self.segments = ['S']
        elif segment_type == 'gaze':
            self.segments = ['G']
        elif segment_type == 'tongue':
            self.segments = ['E030', 'E047', 'E048', 'E049', 'E050', 'E051', 'E052', 'E053', 'E054', 'E055']
        elif segment_type == 'teeth':
            self.segments = ['E029', 'E043', 'E044', 'E054', 'E055']
        else:
            print(f'Unsupported segment type: {segment_type}')
            quit()

    def segmentfilter(self, name):
        if len(self.segments) == 0:
            return True
        for n in self.segments:
            if name.startswith(n):
                return True
        return False

    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_outputlist(self): return ["irgbrec"]
    def get_ae_args(self): return dict(renderoptions={**get_renderoptions(), **{"colorprims": self.colorprims}})
    def get_test_dataset(self): return get_dataset(start_frac = _end_frac+1e-3, end_frac = 1.0)
    def get_dataset(self, train_dataset):
        import data.utils
        import eval.cameras.rotate as cameralib
        train_dataset.datasets[self.ididx].filter_segments_and_camera(self.segmentfilter, self.cam)
        train_dataset.datasets[self.ididx].subsampletype = None
        dataset = datamodel.DatasetMultiSingle(self.ididx, train_dataset)
        if self.cam is None:
            camdataset = cameralib.Dataset(len(dataset), camdist=self.camdist, period=self.camperiod, revs=self.camrevs)
            return data.utils.ColCatDataset(camdataset, dataset)
        else:
            return dataset



class RenderMobile():
    def __init__(self,
                 output_ididx = [0,1,2,3,4],
                 cam=["400029", "400050"],
                 camdist=768., camperiod=128, camrevs=0.25,
                 segment_type='expressions', #'rom', 'expressions', 'gaze', 'tongue', 'speech' #segments=['E0'],#['ROM'],
                 maxframes=-1, relativecamera=True,
                 drawmesh=False, viewtemplate=False, colorprims=False,
                 keyfilter=[]):
        #self.outfilename = outfilename
        #self.outfilesuffix = outfilesuffix

        self.set_segment(segment_type)
        #self.set_identities(id_type, ididx)


        self.cam = cam
        self.camdist = camdist
        self.camperiod = camperiod
        self.camrevs = camrevs


        self.maxframes = maxframes
        self.relativecamera = relativecamera
        self.drawmesh = drawmesh
        self.viewtemplate = viewtemplate
        self.colorprims = colorprims
        self.keyfilter = keyfilter
        self.batchsize = 1

        self.label = False



        self.eval_geofile = ["/mnt/home/jsaragih/code/codec_multi/train/YaserNeutralGeo2.obj",
                             '/mnt/home/chen/CARE/runs/avatar_wild_phone_old_4/avatar_wild_geo_refine/002643814_20210319/mesh.obj',
                             "/mnt/home/jsaragih/code/codec_multi/train/DaniNeutralGeo.obj",
                             "/mnt/home/chen/CARE/runs/avatar_wild_phone_shugao/avatar_wild_geo_refine/002643814_20210319/mesh.obj",
                             "/mnt/home/chen/CARE/runs/avatar_wild_phone_xiaomin/avatar_wild_geo_refine/002643814_20210319/mesh.obj"]
        self.eval_texfile = ["/mnt/home/jsaragih/code/codec_multi/train/YaserNeutralTex2.png",
                             '/mnt/home/chen/CARE/runs/avatar_wild_phone_old_4/avatar_wild_geo_refine/002643814_20210319/tex.png',
                             "/mnt/home/jsaragih/code/codec_multi/train/DaniNeutralTex.png",
                             "/mnt/home/chen/CARE/runs/avatar_wild_phone_shugao/avatar_wild_geo_refine/002643814_20210319/tex.png",
                             "/mnt/home/chen/CARE/runs/avatar_wild_phone_xiaomin/avatar_wild_geo_refine/002643814_20210319/tex.png"]
        self.output_ididx = output_ididx
        self.eval_ididx = output_ididx

        self.imgdir = self.get_imgdir()
        self.vidfile = self.get_vidfile()

    def label_images(self): return self.label


    def get_nframes(self): return 300
    def get_identities(self): return 0, self.output_ididx

    def get_vidfile(self): return os.path.dirname(__file__) + f'/render_mobile.mp4'
    def get_imgdir(self):  return os.path.dirname(__file__) + f'/render_mobile'

    def set_segment(self, segment_type):
        self.segment_type = segment_type
        if segment_type == 'rom':
            self.segments = ['ROM']
        elif segment_type == 'expressions':
            self.segments = ['E0']
        elif segment_type == 'speech':
            self.segments = ['S']
        elif segment_type == 'gaze':
            self.segments = ['G']
        elif segment_type == 'tongue':
            self.segments = ['E030', 'E047', 'E048', 'E049', 'E050', 'E051', 'E052', 'E053', 'E054', 'E055']
        elif segment_type == 'teeth':
            self.segments = ['E029', 'E043', 'E044', 'E054', 'E055']
        else:
            print(f'Unsupported segment type: {segment_type}')
            quit()

    def segmentfilter(self, name):
        if len(self.segments) == 0:
            return True
        for n in self.segments:
            if name.startswith(n):
                return True
        return False

    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_outputlist(self): return ["irgbrec"]
    def get_ae_args(self): return dict(renderoptions={**get_renderoptions(), **{"colorprims": self.colorprims}})
    def get_test_dataset(self): return get_dataset(start_frac = _end_frac+1e-3, end_frac = 1.0)
    def get_dataset(self, train_dataset):
        import data.utils
        import eval.cameras.rotate as cameralib
        train_dataset.datasets[self.ididx].filter_segments_and_camera(self.segmentfilter, self.cam)
        train_dataset.datasets[self.ididx].subsampletype = None
        dataset = datamodel.DatasetMultiSingle(self.ididx, train_dataset)
        if self.cam is None:
            camdataset = cameralib.Dataset(len(dataset), camdist=self.camdist, period=self.camperiod, revs=self.camrevs)
            return data.utils.ColCatDataset(camdataset, dataset)
        else:
            return dataset



#only render a single subject with pre-specified
class RenderMobileSingle():
    def __init__(self,
                 input_ididx = 1, #index of training identity to drive the animation
                 eval_geofile = "/mnt/home/jsaragih/code/codec_multi/train/YaserNeutralGeo2.obj",
                 eval_texfile = "/mnt/home/jsaragih/code/codec_multi/train/YaserNeutralTex2.png",
                 segments=['ROM07_Facial_Expressions'], #segment used to animate results
                 outpath=os.path.dirname(__file__), #path to save animation to
                 cam=None, camdist=768., camperiod=512, camrevs=1.0,
                 maxframes=-1, relativecamera=True,
                 drawmesh=False, viewtemplate=False, colorprims=False,
                 keyfilter=[]):
        #self.outfilename = outfilename
        #self.outfilesuffix = outfilesuffix

        self.input_ididx = input_ididx

        #self.vidfile = outpath + f'/render.mp4'
        self.vidfile = outpath + f'/{segments[0]}.mp4'
        self.imgdir = outpath + f'/{segments[0]}'

        self.cam = cam
        self.camdist = camdist
        self.camperiod = camperiod
        self.camrevs = camrevs
        self.segments = segments
        self.segmentfilter = lambda x: True if len(segments) == 0 else x in segments
        self.maxframes = maxframes
        self.relativecamera = relativecamera
        self.drawmesh = drawmesh
        self.viewtemplate = viewtemplate
        self.colorprims = colorprims
        self.keyfilter = keyfilter

        self.eval_geofile = [eval_geofile]
        self.eval_texfile = [eval_texfile]
        self.eval_ididx = [0]

    def get_identities(self): return self.input_ididx
    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_outputlist(self): return ["irgbrec"]#, "sampledimg"]
    #def get_outputlist(self): return [k[0] if isinstance(k, tuple) else k for k in self.keyfilter]
    def get_ae_args(self): return dict(renderoptions={**get_renderoptions(), **{"colorprims": self.colorprims}})
    def get_test_dataset(self): return get_dataset(start_frac = _end_frac+1e-3, end_frac = 1.0)
    def get_dataset(self, ididx):
        import data.utils
        import eval.cameras.rotate as cameralib
        if self.cam == "all":
            camerafilter = lambda x: x.startswith("40")
        else:
            camerafilter = lambda x: x == self.cam

        #elif self.cam is None:
        #    camerafilter = lambda x: x == "400029"

        dataset = get_dataset(camerafilter=camerafilter,
                              segmentfilter=self.segmentfilter,
                              keyfilter=[k[0] if isinstance(k, tuple) else k for k in self.keyfilter],
                              maxframes=self.maxframes,
                              relativecamera=self.relativecamera,
                              start_frac = 0, end_frac = _end_frac).datasets[ididx]
        if self.cam is None:
            camdataset = cameralib.Dataset(len(dataset), camdist=self.camdist, period=self.camperiod, revs=self.camrevs)
            return data.utils.ColCatDataset(camdataset, dataset)
        else:
            return dataset





class MSE():
    """Evaluate model with training camera or from novel viewpoints.

    e.g., python mse.py {configpath} MSE --maxframes 128"""
    def __init__(self, maxframes=-1, segments=[], cam="all"):
        self.maxframes = maxframes
        self.cam = cam
        self.segments = segments
        self.segmentfilter = lambda x: True if len(segments) == 0 else x in segments
    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_ae_args(self): return dict(outputlist=["irgbrec"], renderoptions=get_renderoptions())
    def get_dataset(self):
        if self.cam == "all":
            camerafilter = lambda x: x.startswith("40")
        elif self.cam == "holdout":
            camerafilter = lambda x: x in holdoutcams
        else:
            camerafilter = lambda x: x == self.cam
        return get_dataset(
            maxframes=self.maxframes,
            camerafilter=camerafilter,
            segmentfilter=self.segmentfilter,
            start_frac = 0, end_frac = _end_frac)


epath=os.getenv('RSC_AVATAR_EVAL_CONFIG_PATH')
exec(open(f"{epath}/eval_config.py").read())

import json

class ObjDict:
    def __init__(self, entries):
        self.add_entries_(entries)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        return self.__dict__.__delitem__(key)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def __getattr__(self, attr):
        if attr.startswith("__"):
            return self.__getattribute__(attr)
        if attr not in self.__dict__:
            self.__dict__[attr] = ObjDict({})
        return self.__dict__[attr]

    def items(self):
        return self.__dict__.items()

    def __iter__(self):
        return iter(self.items())

    def add_entries_(self, entries, overwrite=True):
        for key, value in entries.items():
            if key not in self.__dict__:
                if isinstance(value, dict):
                    self.__dict__[key] = ObjDict(value)
                else:
                    self.__dict__[key] = value
            else:
                if isinstance(value, dict):
                    self.__dict__[key].add_entries_(entries=value, overwrite=overwrite)
                elif overwrite or self.__dict__[key] is None:
                    self.__dict__[key] = value

    def serialize(self):
        return json.dumps(self, default=self.obj_to_dict, indent=4)

    def obj_to_dict(self, obj):
        return obj.__dict__

    def get(self, key, default=None):
        return self.__dict__.get(key, default)
