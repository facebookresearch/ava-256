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


# for change resolution : pass imagesize argument, reset downsample factor, and set subsample size (HEIGHT)
#   in training mode: downsample factor: 2 more --> 1K H by 667 W --> subsample height : 384

import json
import os
from typing import Dict, Set

import numpy as np
import torch

from utils import create_uv_baridx, load_obj


def get_renderoptions():
    return dict(dt=1.0)


def get_autoencoder(dataset, assetpath: str):
    import torch
    import torch.nn as nn

    import models.autoencoder as aemodel
    import models.bg.mlp2d as bglib
    import models.bottlenecks.vae as vae
    import models.colorcals.colorcal as colorcalib
    import models.decoders.assembler as decoderlib
    import models.encoders.expression as expression_encoder_lib
    import models.encoders.identity as identity_encoder_lib
    import models.raymarchers.mvpraymarcher as raymarcherlib

    allcameras = dataset.get_allcameras()
    ncams = len(allcameras)
    width, height = dataset.get_img_size()

    print("@@@ Get autoencoder ABLATION CONFIG FILE : lenth of data set : {}".format(len(dataset.identities)))

    colorcal = colorcalib.Colorcal(len(dataset.get_allcameras()), len(dataset.identities))
    objpath = f"{assetpath}/geotextop.obj"

    dotobj = load_obj(objpath)
    vt, vi, vti = dotobj["vt"], dotobj["vi"], dotobj["vti"]
    vt = np.array(vt, dtype=np.float32)
    vi = np.array(vi, dtype=np.int32)
    vti = np.array(vti, dtype=np.int32)

    print(f"dataset vertmean: {dataset.vertmean.shape}")

    vertmean = torch.from_numpy(dataset.vertmean)
    vertstd = dataset.vertstd

    # load per-textel triangulation indices
    resolution = 1024
    geofile = f"{assetpath}/retop.obj"
    uvpath = f"{assetpath}/fd-data/"
    trifile = f"{uvpath}/uv_tri_{resolution}_orig.txt"
    barfiles = [f"{uvpath}/uv_bary{i}_{resolution}_orig.txt" for i in range(3)]
    uvdata = create_uv_baridx(geofile, trifile, barfiles)

    # Encoders
    expression_encoder = expression_encoder_lib.ExpressionEncoder(uvdata["uv_idx"], uvdata["uv_bary"])
    id_encoder = identity_encoder_lib.IdentityEncoder(uvdata["uv_idx"], uvdata["uv_bary"], wsize=128)

    # VAE bottleneck for the expression encoder
    bottleneck = vae.VAE_bottleneck(64, 16)

    # Decoder
    volradius = 256.0
    decoder = decoderlib.DecoderAssembler(
        vt,
        vi,
        vti,
        vertmean=vertmean,
        vertstd=vertstd,
        volradius=volradius,
        nprims=128 * 128,
        primsize=(8, 8, 8),
    )

    # NOTE(julieta) this ray marcher expects the channels of the template to be last by default
    raymarcher = raymarcherlib.Raymarcher(volradius)

    bgmodel = bglib.BackgroundModelSimple(len(allcameras), len(dataset.identities))
    ae = aemodel.Autoencoder(
        identity_encoder=id_encoder,
        expression_encoder=expression_encoder,
        bottleneck=bottleneck,
        decoder_assembler=decoder,
        raymarcher=raymarcher,
        colorcal=colorcal,
        bgmodel=bgmodel,
    )

    # DO NOT RUN VGG AT ALL and remove vgg in loss_weight for ABLATION TEST : @@@@
    print("id_encoder params:", sum(p.numel() for p in ae.id_encoder.parameters() if p.requires_grad))
    print(f"encoder params: {sum(p.numel() for p in ae.expr_encoder.parameters() if p.requires_grad):_}")
    print(f"decoder params: {sum(p.numel() for p in ae.decoder_assembler.parameters() if p.requires_grad):_}")
    print(f"colorcal params: {sum(p.numel() for p in ae.colorcal.parameters() if p.requires_grad):_}")
    print(f"bgmodel params: {sum(p.numel() for p in ae.bgmodel.parameters() if p.requires_grad):_}")
    print(f"total params: {sum(p.numel() for p in ae.parameters() if p.requires_grad):_}")

    return ae


# profiles
class Train:
    def get_autoencoder(self, dataset, assetpath: str = "/checkpoint/avatar/jinkyuk/rsc-assets"):
        return get_autoencoder(dataset, assetpath)

    def get_output_set(self) -> Set[str]:
        return set(["irgbrec", "sampledimg", "primscale"])

    def get_ae_args(self):
        return dict(renderoptions=get_renderoptions())

    def get_loss_weights(self) -> Dict[str, float]:
        return {"irgbl1": 1.0, "vertl1": 0.1, "kldiv": 0.001, "primvolsum": 0.01}


class ProgressWriterNoBG:
    def batch(self, iternum, itemnum, imagename=None, **kwargs):
        import numpy as np
        from PIL import Image

        rows = []
        row = []
        for i in range(kwargs["image"].size(0)):
            if ("irgbsqerr" in kwargs) and ("sampledimg" in kwargs):
                row.append(
                    np.concatenate(
                        (
                            (
                                torch.sqrt(kwargs["irgbsqerr"][i].data + 1e-3).to("cpu").numpy().transpose((1, 2, 0))
                                * 100.0
                            ).clip(0, 255),
                            kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0)),
                            kwargs["sampledimg"][i].data.to("cpu").numpy().transpose((1, 2, 0)),
                        ),
                        axis=1,
                    )
                )
            else:
                row.append(
                    np.concatenate(
                        (
                            kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                            kwargs["image"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                        ),
                        axis=1,
                    )
                )
            if len(row) == 6:
                rows.append(np.concatenate(row, axis=1))
                row = []
        imgout = np.concatenate(rows, axis=0) * 2.0
        outpath = os.path.dirname(__file__)
        if imagename is None:
            Image.fromarray(np.clip(imgout, 0, 255).astype(np.uint8)).save(
                os.path.join(outpath, "prog_{:06}.jpg".format(iternum))
            )
        else:
            Image.fromarray(np.clip(imgout, 0, 255).astype(np.uint8)).save(os.path.join(outpath, imagename))

    def finalize(self):
        pass


class ProgressWriter:
    def batch(self, iternum, itemnum, imagename=None, **kwargs):
        import numpy as np
        from PIL import Image

        rows = []
        row = []

        print(
            " KWARGS IMAGE SIZE (0) : {} = imagename {} -- kwargs.keys(): {}".format(
                kwargs["image"].size(0), imagename, kwargs.keys()
            )
        )

        for i in range(kwargs["image"].size(0)):
            if ("irgbsqerr" in kwargs) and ("sampledimg" in kwargs):  # training output
                if "bg" in kwargs and kwargs["bg"] != None:
                    row.append(
                        np.concatenate(
                            (
                                (
                                    torch.sqrt(kwargs["irgbsqerr"][i].data + 1e-3)
                                    .to("cpu")
                                    .numpy()
                                    .transpose((1, 2, 0))
                                    * 100.0
                                ).clip(0, 255),
                                kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0)),
                                kwargs["bg"][i].data.to("cpu").numpy().transpose((1, 2, 0)),
                                kwargs["sampledimg"][i].data.to("cpu").numpy().transpose((1, 2, 0)),
                            ),
                            axis=1,
                        )
                    )
                else:
                    row.append(
                        np.concatenate(
                            (
                                (
                                    torch.sqrt(kwargs["irgbsqerr"][i].data + 1e-3)
                                    .to("cpu")
                                    .numpy()
                                    .transpose((1, 2, 0))
                                    * 100.0
                                ).clip(0, 255),
                                kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0)),
                                kwargs["sampledimg"][i].data.to("cpu").numpy().transpose((1, 2, 0)),
                            ),
                            axis=1,
                        )
                    )
            else:  # testoutputs
                if "bg" in kwargs and kwargs["bg"] != None:
                    row.append(
                        np.concatenate(
                            (
                                kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                                kwargs["bg"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                                kwargs["image"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                            ),
                            axis=1,
                        )
                    )
                else:
                    row.append(
                        np.concatenate(
                            (
                                kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                                kwargs["image"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                            ),
                            axis=1,
                        )
                    )

            if len(row) == 4:
                rows.append(np.concatenate(row, axis=1))
                row = []

        print(" len(ROWS) : {}".format(len(rows)))
        if len(rows) == 0:
            rows.append(np.concatenate(row, axis=1))

        imgout = np.concatenate(rows, axis=0)  # * 2.

        outpath = os.path.dirname(__file__)
        print("OUTPATH {} -- imagename {}".format(outpath, imagename))

        if imagename is None:
            Image.fromarray(np.clip(imgout, 0, 255).astype(np.uint8)).save(
                os.path.join(outpath, "prog_{:06}.png".format(iternum))
            )
        else:
            Image.fromarray(np.clip(imgout, 0, 255).astype(np.uint8)).save(os.path.join(outpath, imagename))

    def finalize(self):
        pass


# TODO: option to control frequency of output
class Progress:
    batchsize = 4

    def get_output_set(self) -> Set[str]:
        return set(["irgbrec", "bg"])

    def get_ae_args(self):
        return dict(renderoptions=get_renderoptions())

    # def get_dataset(self): return get_dataset(start_frac = 0, end_frac = 1.0)
    def get_dataset(self, mds, ids):
        return get_dataset(mds, ids)

    def get_writer(self):
        return ProgressWriter()


class Render:
    def __init__(
        self,
        input_ididx=1,
        output_ididx=[1, 0, -1, -2, -3],  # [12,13,14,15,16,17,18,19], #[4, 5, 6, 7, 8, 9, 10, 11], #
        test_output_ididx=[0, 1, -1, -2, -3],  # [-5, -6, -7, -8, -9, -10, -11, -12], #
        # outfilesuffix=None,
        cam=None,
        camdist=768.0,
        camperiod=128,
        camrevs=0.5,  # 0.25,
        segments=["ROM07_Facial_Expressions"],
        # segments=['S9_They_had_slapped_their_thighs._'],#segments=["S01_She_always_jokes_about_too_much_garlic_in_his_food"],
        maxframes=-1,
        relativecamera=True,
        drawmesh=False,
        viewtemplate=False,
        colorprims=False,
        keyfilter=[],
    ):
        # self.outfilename = outfilename
        # self.outfilesuffix = outfilesuffix

        self.input_ididx = input_ididx
        self.output_ididx = output_ididx
        self.test_output_ididx = test_output_ididx

        self.vidfile = os.path.dirname(__file__) + "/render.mp4"
        self.imgdir = os.path.dirname(__file__) + "/render_images"

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

    def get_identities(self):
        return self.input_ididx, self.output_ididx, self.test_output_ididx

    def get_autoencoder(self, dataset):
        return get_autoencoder(dataset)

    def get_outputlist(self):
        return ["irgbrec"]

    # def get_outputlist(self): return [k[0] if isinstance(k, tuple) else k for k in self.keyfilter]
    def get_ae_args(self):
        return dict(renderoptions={**get_renderoptions(), **{"colorprims": self.colorprims}})

    # def get_test_dataset(self): return get_dataset(start_frac = _end_frac+1e-3, end_frac = 1.0)
    def get_test_dataset(self):
        return get_dataset(start_frac=_test_start_frac, _test_end_frac=1.0)

    def get_dataset(self, ididx):
        import eval.cameras.rotate as cameralib

        import data.utils

        if self.cam == "all":
            camerafilter = lambda x: x.startswith("40")
        else:
            camerafilter = lambda x: x == self.cam

        # elif self.cam is None:
        #    camerafilter = lambda x: x == "400029"

        dataset = get_dataset(
            camerafilter=camerafilter,
            segmentfilter=self.segmentfilter,
            keyfilter=[k[0] if isinstance(k, tuple) else k for k in self.keyfilter],
            maxframes=self.maxframes,
            relativecamera=self.relativecamera,
            start_frac=0,
            end_frac=_end_frac,
        ).datasets[ididx]
        if self.cam is None:
            camdataset = cameralib.Dataset(len(dataset), camdist=self.camdist, period=self.camperiod, revs=self.camrevs)
            return data.utils.ColCatDataset(camdataset, dataset)
        else:
            return dataset


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
