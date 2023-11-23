"""
Volumetric autoencoder (image -> encoding -> volume -> image)
"""
import inspect
import time
from typing import Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import models.utils
from extensions.computeraydirs.computeraydirs import compute_raydirs

# FLAGS to profile times
AETIME = "aetime"  # total autoencoder time
IDENCTIME = "idenctime"  # id encoding time
ENCTIME = "enctime"  # expression encoding time
DECTIME = "dectime"  # decoding time
RAYMARCHINGTIME = "rmtime"  # ray marching time
VERTLOSSTIME = "vertlosstime"  # geometry loss time
RGBLOSSTIME = "rgblosstime"  # image loss time
COLORCALANDBGTIME = "colorcalandbg"  # color calibration and background model


def color_normalize(src, dst):
    b, h, w = src.shape[0], src.shape[-2], src.shape[-1]
    A = src.view(-1, 3, w * h)
    B = dst.view(-1, 3, w * h)

    # mean normalize
    Amean = torch.mean(A, dim=-1, keepdims=True)
    Bmean = torch.mean(B, dim=-1, keepdims=True)
    A = A - Amean
    B = B - Bmean
    # AAt = torch.bmm(A, A.permute(0,2,1))
    # BAt = torch.bmm(B, A.permute(0,2,1))
    AAt = A @ A.permute(0, 2, 1)
    BAt = B @ A.permute(0, 2, 1)
    for i in range(3):
        AAt[:, i, i] += 1e-3
    AAti = torch.inverse(AAt)
    x = torch.bmm(BAt, AAti)
    C = torch.bmm(x, A) + Bmean

    out = C.view(-1, 3, h, w)

    return out


class Autoencoder(nn.Module):
    def __init__(
        self,
        dataset: torch.data.Dataset,
        id_encoder: torch.nn.Module,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        raymarcher: torch.nn.Module,
        colorcal: torch.nn.Module,
        bgmodel: torch.nn.Module,
        encoderinputs,
        topology=None,
        imagemean=0.0,
        imagestd=1.0,
        use_vgg=False,
        use_id_latents=False,
    ):
        super(Autoencoder, self).__init__()

        self.id_encoder = id_encoder
        self.encoder = encoder
        self.decoder = decoder
        self.raymarcher = raymarcher
        self.colorcal = colorcal
        self.bgmodel = bgmodel
        self.encoderinputs = encoderinputs
        self.use_id_latents = use_id_latents  # whether to have trainable latents

        # renderlayer
        if topology is not None:
            self.vt, self.vi, self.vti = topology["vt"], topology["vi"], topology["vti"]

        ## TODO: delay evaluating RenderLayer until forward
        # self.width, self.height = next(iter(dataset.().values()))["size"]
        ##self.width, self.height = 667, 1024
        # self.rl = RenderLayer(maxbatch, self.height, self.width, vt, vi, vti)

        # TODO: would be best if this is transparent to autoencoder/passed to
        # decoder in a dict, but standard loss is in normalized space
        # self.vertmean = torch.from_numpy(dataset.vertmean)
        # self.texmean = torch.from_numpy(dataset.texmean)
        # self.texstd = dataset.texstd
        self.register_buffer("vertmean", torch.from_numpy(dataset.vertmean), persistent=False)
        self.vertstd = dataset.vertstd
        self.register_buffer("texmean", torch.from_numpy(dataset.texmean), persistent=False)
        self.texstd = dataset.texstd
        self.imagemean = imagemean
        self.imagestd = imagestd

        if use_vgg:
            self.net_vgg = VGGLossMasked2()
            # print('VGG Not supported yet!')
            # quit()
        else:
            self.net_vgg = None

        # Create backpropagable latent texture for the id encoder
        if self.use_id_latents:
            # List of `MugsyCapture`s
            self.captures = dataset.identities

            neut_avgtex = dict()
            for capture, dataset in dataset.single_capture_datasets.items():
                neut_avgtex[str(capture)] = nn.Parameter(torch.Tensor(dataset.neut_avgtex.copy()))
            self.neut_avgtex = nn.ParameterDict(neut_avgtex)

    def id_encode(self, neut_verts, neut_avgtex, **kwargs):
        if (neut_verts is None) or (neut_avgtex is None):
            print(f"Empty identity conditioning data")
            quit()
        id_cond = self.id_encoder(neut_verts, neut_avgtex)[0]  # {z_tex_id, b_tex_id, z_geo_id, b_geo_id}
        return id_cond

    # @profile
    def forward(
        self,
        camrot: torch.Tensor,
        campos: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
        bgcolor: Optional[torch.Tensor] = None,
        relheadrot: Optional[torch.Tensor] = None,
        gaze: Optional[torch.Tensor] = None,
        idindex: Optional[torch.Tensor] = None,
        camindex: Optional[torch.Tensor] = None,
        pixelcoords: Optional[torch.Tensor] = None,
        modelmatrix: Optional[torch.Tensor] = None,
        validinput: Optional[torch.Tensor] = None,
        avgtex: Optional[torch.Tensor] = None,
        verts: Optional[torch.Tensor] = None,
        neut_avgtex: Optional[torch.Tensor] = None,
        neut_verts: Optional[torch.Tensor] = None,
        cond_avgtex: Optional[torch.Tensor] = None,
        cond_verts: Optional[torch.Tensor] = None,
        fixedcamimage: Optional[torch.Tensor] = None,
        encoding: Optional[torch.Tensor] = None,
        id_cond: Optional[dict] = None,
        image: Optional[torch.Tensor] = None,
        imagemask: Optional[torch.Tensor] = None,
        imagevalid: Optional[torch.Tensor] = None,
        bg: Optional[torch.Tensor] = None,
        segmentation: Optional[torch.Tensor] = None,
        renderoptions: dict = {},
        trainiter: int = -1,
        outputlist: list = [],
        losslist: list = [],
        **kwargs,
    ):
        """
        Parameters
        ----------
        camrot : torch.Tensor [B, 3, 3]
            Rotation matrix of target view camera
        campos : torch.Tensor [B, 3]
            Position of target view camera
        focal : torch.Tensor [B, 2]
            Focal length of target view camera
        princpt : torch.Tensor [B, 2]
            Princple point of target view camera
        camindex : torch.Tensor[int32], optional [B]
            Camera index within the list of all cameras
        pixelcoords : torch.Tensor, optional [B, H', W', 2]
            Pixel coordinates to render of the target view camera
        modelmatrix : torch.Tensor, optional [B, 3, 3]
            Relative transform from the 'neutral' pose of object
        validinput : torch.Tensor, optional [B]
            Whether the current batch element is valid (used for missing images)
        avgtex : torch.Tensor, optional [B, 3, 1024, 1024]
            Texture map averaged from all viewpoints
        verts : torch.Tensor, optional [B, 7306, 3]
            Mesh vertex positions
        fixedcamimage : torch.Tensor, optional [B, 3, 512, 334]
            Camera images from a one or more cameras that are always the same
            (i.e., unrelated to target)
        encoding : torch.Tensor, optional [B, 256]
            Direct encodings (overrides encoder)
        image : torch.Tensor, optional [B, 3, H, W]
            Target image
        imagemask : torch.Tensor, optional [B, 1, H, W]
            Target image mask for computing reconstruction loss
        imagevalid : torch.Tensor, optional [B]
        bg : torch.Tensor, optional [B, 3, H, W]
        renderoptions : dict
            Rendering/raymarching options (e.g., stepsize, whether to output debug images, etc.)
        trainiter : int
            Training iteration number
        outputlist : list
            What values to return (e.g., image reconstruction, debug output)
        losslist : list
            What losses to output (e.g., image reconstruction loss, priors)

        Returns
        -------
        result : dict
            Contains items specified in outputlist

        """

        if self.use_id_latents:
            N, C, H, W = neut_avgtex.shape

            # Collect all the latents and convert to tensor, should backprop to it
            neut_avgtex = []
            for capture_id in idindex:
                capture = self.captures[capture_id]
                normed = (self.neut_avgtex[str(capture)] - self.texmean) / self.texstd
                neut_avgtex.append(normed)

            neut_avgtex = torch.stack(neut_avgtex)

        resultout = {}
        resultlosses = {}

        if AETIME in outputlist:
            torch.cuda.synchronize()
            aestart = time.time()

        if IDENCTIME in outputlist:
            torch.cuda.synchronize()
            idencstart = time.time()

        # get identity conditioning
        if (neut_verts is None) or (neut_avgtex is None):
            raise ValueError(f"Empty identity conditioning data")

        if id_cond is None and self.id_encoder is not None:
            if (cond_avgtex is None) or (cond_verts is None):
                randomize_color = False
                if "randomize_color" in kwargs:
                    randomize_color = kwargs.get("randomize_color")
                if self.training and randomize_color:
                    assert False  # no randomize color

                    with torch.no_grad():
                        n = neut_avgtex.shape[0]
                        neut_avgtex_pert = []
                        for i in range(n):
                            # R, _ = cvRodrigues(np.random.randn(3) * np.pi / 6)
                            R = None

                            scale = np.random.randn(1) * 0.1 + 1.0
                            R = torch.from_numpy((scale * R).astype(np.float32)).cuda()
                            im = neut_avgtex[i] * self.texstd + self.texmean
                            im = R @ im.view(3, -1)
                            im = (im.view(3, 1024, 1024) - self.texmean) / self.texstd
                            neut_avgtex_pert.append(im.unsqueeze(0))
                        neut_avgtex_pert = torch.cat(neut_avgtex_pert)

                    id_cond, idenc_losses = self.id_encoder(neut_verts, neut_avgtex_pert, losslist=losslist)
                else:
                    normalize_color = False
                    if "normalize_color" in kwargs:
                        normalize_color = kwargs.get("normalize_color")
                    if self.training and normalize_color:
                        with torch.no_grad():
                            dst = self.texmean.unsqueeze(0)
                            neut_avgtex_pert = (
                                color_normalize(neut_avgtex * self.texstd + dst, dst) - dst
                            ) / self.texstd

                        id_cond, idenc_losses = self.id_encoder(neut_verts, neut_avgtex_pert, losslist=losslist)
                    else:
                        id_cond, idenc_losses = self.id_encoder(
                            neut_verts, neut_avgtex, losslist=losslist
                        )  # {z_tex_id, b_tex_id, z_geo_id, b_geo_id}

            else:
                id_cond, idenc_losses = self.id_encoder(
                    neut_verts, neut_avgtex, cond_verts, cond_avgtex, losslist=losslist
                )  # {z_tex_id, b_tex_id, z_geo_id, b_geo_id}

            if idenc_losses is not None:
                resultlosses.update(idenc_losses)
        else:
            idenc_losses = None

        # zero-out identity biases
        if "zerobias" in kwargs:
            zero_out_biases = kwargs.get("zerobias")
            with torch.no_grad():
                if zero_out_biases:
                    for i in range(len(id_cond["b_geo"])):
                        id_cond["b_geo"][i] = (id_cond["b_geo"][i] * 0).detach()
                    for i in range(len(id_cond["b_tex"])):
                        id_cond["b_tex"][i] = (id_cond["b_tex"][i] * 0).detach()

        if "id_cond" in outputlist:
            resultout["id_cond"] = id_cond

        if IDENCTIME in outputlist:
            torch.cuda.synchronize()
            resultout[IDENCTIME] = time.time() - idencstart

        if ENCTIME in outputlist:
            torch.cuda.synchronize()
            encstart = time.time()

        # 0. encode/get encoding
        gazepred, neckrotpred = None, None
        if encoding is None:
            # TODO: fixed potential set of inputs (e.g., what about time)

            encout, enclosses = self.encoder(
                *[
                    dict(
                        verts=verts,
                        avgtex=avgtex,
                        fixedcamimage=fixedcamimage,
                        neut_verts=neut_verts,
                        neut_avgtex=neut_avgtex,
                        gaze=gaze,
                    )[k]
                    for k in self.encoderinputs
                ],
                losslist=losslist,
            )

            encoding = encout["encoding"]
            # result["losses"].update(encout["losses"])
            resultlosses.update(enclosses)
            resultout["encoding"] = encoding

            if "gaze" in encout:
                gazepred = encout["gaze"]
                resultout["gazepred"] = gazepred

            if "neckrot" in encout:
                neckrotpred = encout["neckrot"]
                resultout["neckrotpred"] = neckrotpred

        if ENCTIME in outputlist:
            torch.cuda.synchronize()
            encend = time.time()
            resultout[ENCTIME] = encend - encstart

        # 1. decode vol/mesh
        if DECTIME in outputlist:
            torch.cuda.synchronize()
            decstart = time.time()

        # compute relative viewing position
        viewrot = torch.bmm(camrot, modelmatrix[:, :3, :3])
        viewpos = torch.bmm((campos[:, :] - modelmatrix[:, :3, 3])[:, None, :], modelmatrix[:, :3, :3])[:, 0, :]
        viewdir = viewpos / torch.sqrt(torch.sum(viewpos**2, dim=-1, keepdim=True))

        condinput = []
        if gazepred is not None:
            condinput.append(gazepred)
        if neckrotpred is not None:
            condinput.append(neckrotpred)
        if len(condinput) > 0:
            condinput = torch.cat(condinput, dim=-1)
            decout, declosses = self.decoder(
                verts,
                id_cond,
                encoding,
                viewpos,
                #  condinput=condinput,
                renderoptions=renderoptions,
                trainiter=trainiter,
                losslist=losslist,
            )
        else:
            # id_cond_dummy = torch.randn((4, 1))
            decout, declosses = self.decoder(
                verts,
                id_cond,
                encoding,
                viewpos,
                #  condinput=condinput,
                renderoptions=renderoptions,
                trainiter=trainiter,
                losslist=losslist,
            )

        resultlosses.update(declosses)

        if DECTIME in outputlist:
            torch.cuda.synchronize()
            decend = time.time()
            resultout[DECTIME] = decend - decstart

        if VERTLOSSTIME in outputlist:
            torch.cuda.synchronize()
            vertlossstart = time.time()

        # compute vert loss
        if "vertmse" in losslist:
            weight = validinput[:, None, None].expand_as(verts)

            vertsrecstd = (decout["verts"] - self.vertmean) / self.vertstd

            if False:  # self.vertl1: #tmp
                vertsqerr = weight * torch.abs(verts - vertsrecstd)
            else:
                vertsqerr = weight * (verts - vertsrecstd) ** 2

            vertmse = torch.sum(vertsqerr.view(vertsqerr.size(0), -1), dim=-1)
            vertmse_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)

            # result["losses"]["vertmse"] = (vertmse, vertmse_weight)
            resultlosses["vertmse"] = (vertmse, vertmse_weight)

        if "vertl1" in losslist:
            weight = validinput[:, None, None].expand_as(verts)

            vertsrecstd = (decout["verts"] - self.vertmean) / self.vertstd

            vertsqerr = weight * torch.abs(verts - vertsrecstd)

            vertmse = torch.sum(vertsqerr.view(vertsqerr.size(0), -1), dim=-1)
            vertmse_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)

            resultlosses["vertl1"] = (vertmse, vertmse_weight)

        if "vertl1_mod" in losslist:
            if "verts_mask" not in decout:
                print(f"verts_mask decoder output required for vertl1_mod loss")
                quit()
            if "verts_gt" not in decout:
                print(f"verts_gt decoder output required for vertl1_mod loss")
                quit()
            if "verts_pred" not in decout:
                print(f"verts_pred decoder output required for vertl1_mod loss")
                quit()
            weight = validinput[:, None, None].expand_as(decout["verts_gt"]) * decout["verts_mask"]
            vertsqerr = weight * torch.abs(decout["verts_gt"] - decout["verts_pred"])
            vertmse = torch.sum(vertsqerr.view(vertsqerr.size(0), -1), dim=-1)
            vertmse_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)
            resultlosses["vertl1_mod"] = (vertmse, vertmse_weight)

        # TODO: check this
        # subsample depth, imagerec, imagerecmask
        if image is not None and pixelcoords.size()[1:3] != image.size()[2:4]:
            samplecoords = torch.cat(
                [
                    pixelcoords[..., :1] * 2 / (image.shape[-1] - 1) - 1,
                    pixelcoords[..., 1:] * 2 / (image.shape[-2] - 1) - 1,
                ],
                dim=-1,
            )
        else:
            samplecoords = torch.cat(
                [
                    pixelcoords[..., :1] * 2 / (pixelcoords.shape[-2] - 1) - 1,
                    pixelcoords[..., 1:] * 2 / (pixelcoords.shape[-3] - 1) - 1,
                ],
                dim=-1,
            )

        if "samplecoords" in outputlist:
            resultout["samplecoords"] = samplecoords

        # NOTE(julieta) I know... we are including the time to compute sample coords inside vertlosstime
        # I do not want to create another timer, but I want to account for every line
        if VERTLOSSTIME in outputlist:
            torch.cuda.synchronize()
            resultout[VERTLOSSTIME] = time.time() - vertlossstart

        # 2. raymarch
        if RAYMARCHINGTIME in outputlist:
            torch.cuda.synchronize()
            rmstart = time.time()

        # compute rays TODO: encapsulate?
        # NHWC
        raydir = (pixelcoords - princpt[:, None, None, :]) / focal[:, None, None, :]
        raydir = torch.cat([raydir, torch.ones_like(raydir[:, :, :, 0:1])], dim=-1)
        raydir = torch.sum(viewrot[:, None, None, :, :] * raydir[:, :, :, :, None], dim=-2)
        raydir = raydir / torch.sqrt(torch.sum(raydir**2, dim=-1, keepdim=True))

        raypos, raydir, tminmax = compute_raydirs(
            campos, camrot, focal, princpt, pixelcoords, self.raymarcher.volume_radius
        )
        rayposbeg = campos[:, None, None, :] + raydir * tminmax[:, :, :, :1]
        rayposend = campos[:, None, None, :] + raydir * tminmax[:, :, :, 1:]

        rayrgb, rayalpha, rayrgba, pos_img = self.raymarcher(
            raypos,
            raydir,
            tminmax,
            outputs=decout,
            with_pos_img=("pos_img" in outputlist),
        )
        if "pos_img" in outputlist:
            resultout["pos_img"] = pos_img

        if RAYMARCHINGTIME in outputlist:
            torch.cuda.synchronize()
            rmend = time.time()
            resultout[RAYMARCHINGTIME] = rmend - rmstart

        if COLORCALANDBGTIME in outputlist:
            torch.cuda.synchronize()
            colorcalandbgstart = time.time()

        # Subsample image
        chair = None
        with torch.no_grad():
            if image is not None:
                if pixelcoords.size()[1:3] != image.size()[2:4]:
                    image = F.grid_sample(image, samplecoords, align_corners=True)
                    if imagemask is not None:
                        imagemask = F.grid_sample(imagemask, samplecoords, align_corners=True)
                    if segmentation is not None:
                        segmentation = F.grid_sample(segmentation.float(), samplecoords, align_corners=True)
                else:
                    if segmentation is not None:
                        segmentation = segmentation.float()

                if segmentation is not None:
                    # remove chair
                    chair = 1.0 - (torch.abs(segmentation - 3) < 0.5).float()

                    if imagemask is None:
                        imagemask = chair
                    else:
                        imagemask = imagemask * chair

        # color correction
        if (self.colorcal is None) and (image is not None):
            rayrgb = color_normalize(rayrgb, image * rayalpha)
        else:
            if (camindex is not None) and (idindex is not None):
                rayrgb = self.colorcal(rayrgb, camindex, idindex)

        # 4. bg decode
        if bg is None:
            # TODO(julieta) only call the model if no bg passed in the call

            if "dataset_type" in kwargs:
                dataset_type = kwargs.get("dataset_type")
                bg = self.bgmodel(dataset_type, camindex, idindex, samplecoords)
            else:
                bg = self.bgmodel(camindex, idindex, samplecoords)

        if "bg" in outputlist:
            resultout["bg"] = bg

        # 6. matting
        rastrgb = None
        bgcolor = None
        if rastrgb is not None:
            if bg is not None:
                rayrgb = rayrgb + (1.0 - rayalpha) * (rlirgbrec + (1.0 - rlialpharec) * bg)
            else:
                rayrgb = rayrgb + (1.0 - rayalpha) * rlirgbrec
        else:
            if bgcolor is not None:
                rayrgb = rayrgb + (1.0 - rayalpha) * bgcolor.to("cuda")[None, :, None, None]
            elif bg is not None:
                rayrgb = rayrgb + (1.0 - rayalpha) * bg
            else:
                c = np.asarray([0, 0, 0], dtype=np.float32)
                rayrgb = rayrgb + (1.0 - rayalpha) * torch.from_numpy(c).to("cuda")[None, :, None, None]

        if "irgbrec" in outputlist:
            resultout["irgbrec"] = rayrgb

        if "ialpha" in outputlist:
            resultout["ialpha"] = rayalpha

        if COLORCALANDBGTIME in outputlist:
            torch.cuda.synchronize()
            resultout[COLORCALANDBGTIME] = time.time() - colorcalandbgstart

        if RGBLOSSTIME in outputlist:
            torch.cuda.synchronize()
            rgblossstart = time.time()

        # irgb loss
        if image is not None:
            if "sampledimg" in outputlist:
                resultout["sampledimg"] = image

            # standardize
            rayrgb = (rayrgb - self.imagemean) / self.imagestd
            image = (image - self.imagemean) / self.imagestd

            # compute reconstruction loss weighting
            weight = torch.ones_like(image) * validinput[:, None, None, None]
            if imagevalid is not None:
                weight = weight * imagevalid[:, None, None, None]
            if imagemask is not None:
                weight = weight * imagemask

            if chair is not None:
                weight = weight * chair

            if "imageweight" in outputlist:
                resultout["imageweight"] = weight

            irgbsqerr = (weight * (image - rayrgb) ** 2).contiguous()

            if "irgbsqerr" in outputlist:
                resultout["irgbsqerr"] = irgbsqerr

            if "irgbmse" in losslist:
                irgbabs = (weight * ((image - rayrgb) ** 2)).contiguous()
                irgbl2 = torch.sum(irgbabs.view(irgbabs.size(0), -1), dim=-1)
                irgbl2_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)
                resultlosses["irgbmse"] = (irgbl2, irgbl2_weight.clamp(min=1e-6))

            if "irgbl1" in losslist:
                irgbabs = (weight * torch.abs(image - rayrgb)).contiguous()
                irgbl1 = torch.sum(irgbabs.view(irgbabs.size(0), -1), dim=-1)
                irgbl1_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)
                resultlosses["irgbl1"] = (irgbl1, irgbl1_weight.clamp(min=1e-6))

            # TODO(julieta) support VGG loss?

            if "ialphal1" in losslist:
                if segmentation is None:
                    print("Asked for alphal1 loss but no segmentation found")
                    quit()
                with torch.no_grad():
                    alpha = (
                        (torch.abs(segmentation - 1) < 0.5)
                        | (torch.abs(segmentation - 2) < 0.5)
                        | (torch.abs(segmentation - 4) < 0.5)
                    ).float()

                segmask = (torch.sum(segmentation.view(segmentation.shape[0], -1), dim=-1) > 0).float()

                ialphaabs = (chair * torch.abs(alpha - rayalpha)).contiguous() * segmask[:, None, None, None]
                ialphal1 = torch.sum(ialphaabs.view(ialphaabs.size(0), -1), dim=-1)
                ialphal1_weight = torch.sum(chair.view(chair.size(0), -1), dim=-1)
                resultlosses["ialphal1"] = (ialphal1, ialphal1_weight.clamp(min=1e-6))

            if "ialphabce" in losslist:
                if segmentation is None:
                    print("Asked for alphal1 loss but no segmentation found")
                    quit()
                with torch.no_grad():
                    alpha = (
                        (torch.abs(segmentation - 1) < 0.5)
                        | (torch.abs(segmentation - 2) < 0.5)
                        | (torch.abs(segmentation - 4) < 0.5)
                    ).float()

                L = torch.nn.BCELoss(reduction="none")
                ialphaabs = (chair * L(rayalpha, alpha)).contiguous()
                ialphal1 = torch.sum(ialphaabs.view(ialphaabs.size(0), -1), dim=-1)
                ialphal1_weight = torch.sum(chair.view(chair.size(0), -1), dim=-1)
                resultlosses["ialphabce"] = (ialphal1, ialphal1_weight.clamp(min=1e-6))

        if RGBLOSSTIME in outputlist:
            torch.cuda.synchronize()
            resultout[RGBLOSSTIME] = time.time() - rgblossstart

        if "aetime" in outputlist:
            torch.cuda.synchronize()
            resultout["aetime"] = time.time() - aestart

        return resultout, resultlosses
