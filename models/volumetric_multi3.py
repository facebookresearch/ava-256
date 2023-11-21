"""
Volumetric autoencoder (image -> encoding -> volume -> image)
"""
import inspect
import time
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.utils
import einops

from PIL import Image

from extensions.computeraydirs.computeraydirs import compute_raydirs

# FLAGS to profile times
AETIME = "aetime"                # total autoencoder time
IDENCTIME = "idenctime"          # id encoding time
ENCTIME = "enctime"              # expression encoding time
DECTIME = "dectime"              # decoding time
RAYMARCHINGTIME = "rmtime"       # ray marching time
VERTLOSSTIME = "vertlosstime"    # geometry loss time
RGBLOSSTIME = "rgblosstime"      # image loss time
COLORCALANDBGTIME = "colorcalandbg"  # color calibration and background model

def color_normalize(src, dst):

    b, h, w = src.shape[0], src.shape[-2], src.shape[-1]
    A = src.view(-1,3,w*h)
    B = dst.view(-1,3,w*h)

    #mean normalize
    Amean = torch.mean(A, dim=-1, keepdims=True)
    Bmean = torch.mean(B, dim=-1, keepdims=True)
    A = A - Amean
    B = B - Bmean
    #AAt = torch.bmm(A, A.permute(0,2,1))
    #BAt = torch.bmm(B, A.permute(0,2,1))
    AAt = A @ A.permute(0,2,1)
    BAt = B @ A.permute(0,2,1)
    for i in range(3):
        AAt[:,i,i] += 1e-3
    AAti = torch.inverse(AAt)
    x = torch.bmm(BAt, AAti)
    C = torch.bmm(x, A) + Bmean

    out = C.view(-1,3,h,w)

    return out

class Autoencoder(nn.Module):
    def __init__(self, dataset, id_encoder, encoder, decoder, raymarcher, colorcal, bgmodel, encoderinputs,
                 topology=None, imagemean=0., imagestd=1., use_vgg=False, use_id_latents=False):
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
        #self.width, self.height = next(iter(dataset.().values()))["size"]
        ##self.width, self.height = 667, 1024
        #self.rl = RenderLayer(maxbatch, self.height, self.width, vt, vi, vti)

        # TODO: would be best if this is transparent to autoencoder/passed to
        # decoder in a dict, but standard loss is in normalized space
        #self.vertmean = torch.from_numpy(dataset.vertmean)
        #self.texmean = torch.from_numpy(dataset.texmean)
        #self.texstd = dataset.texstd
        self.register_buffer("vertmean", torch.from_numpy(dataset.vertmean), persistent=False)
        self.vertstd = dataset.vertstd
        self.register_buffer("texmean", torch.from_numpy(dataset.texmean), persistent=False)
        self.texstd = dataset.texstd
        self.imagemean = imagemean
        self.imagestd = imagestd

        if use_vgg:
            self.net_vgg = VGGLossMasked2()
            #print('VGG Not supported yet!')
            #quit()
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
            print(f'Empty identity conditioning data')
            quit()
        id_cond = self.id_encoder(neut_verts, neut_avgtex)[0] #{z_tex_id, b_tex_id, z_geo_id, b_geo_id}
        return id_cond


    #@profile
    def forward(self,
                camrot : torch.Tensor,
                campos : torch.Tensor,
                focal : torch.Tensor,
                princpt : torch.Tensor,
                bgcolor : Optional[torch.Tensor] = None,
                relheadrot : Optional[torch.Tensor] = None,
                gaze : Optional[torch.Tensor] = None,
                idindex : Optional[torch.Tensor] = None,
                camindex : Optional[torch.Tensor] = None,
                pixelcoords : Optional[torch.Tensor]=None,
                modelmatrix : Optional[torch.Tensor]=None,
                validinput : Optional[torch.Tensor]=None,
                avgtex : Optional[torch.Tensor]=None,
                verts : Optional[torch.Tensor]=None,
                neut_avgtex : Optional[torch.Tensor]=None,
                neut_verts : Optional[torch.Tensor]=None,
                cond_avgtex : Optional[torch.Tensor]=None,
                cond_verts : Optional[torch.Tensor]=None,
                fixedcamimage : Optional[torch.Tensor]=None,
                encoding : Optional[torch.Tensor]=None,
                id_cond: Optional[dict]=None,
                image : Optional[torch.Tensor]=None,
                imagemask : Optional[torch.Tensor]=None,
                imagevalid : Optional[torch.Tensor]=None,
                bg : Optional[torch.Tensor]=None,
                segmentation : Optional[torch.Tensor]=None,
                renderoptions : dict ={},
                trainiter : int=-1,
                outputlist : list=[],
                losslist : list=[],
                **kwargs):

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



        # inputs={...}, condinputs={...} ?
        resultout = {}#"losses": {}}
        resultlosses = {}


        # print(" TRAIN ITER : {}".format(trainiter))

        #print("enc param", sum([param.nelement()*param.element_size() for param in self.encoder.parameters()]))
        #print("enc buf", sum([buf.nelement()*buf.element_size() for buf in self.encoder.buffers()]))
        #print("dec param", sum([param.nelement()*param.element_size() for param in self.decoder.parameters()]))
        #print("dec buf", sum([buf.nelement()*buf.element_size() for buf in self.decoder.buffers()]))
        #print("rm param", sum([param.nelement()*param.element_size() for param in self.raymarcher.parameters()]))
        #print("rm buf", sum([buf.nelement()*buf.element_size() for buf in self.raymarcher.buffers()]))
        #print("cc param", sum([param.nelement()*param.element_size() for param in self.colorcal.parameters()]))
        #print("cc buf", sum([buf.nelement()*buf.element_size() for buf in self.colorcal.buffers()]))
        #print("bg param", sum([param.nelement()*param.element_size() for param in self.bgmodel.parameters()]))
        #print("bg buf", sum([buf.nelement()*buf.element_size() for buf in self.bgmodel.buffers()]))
        #print("total param", sum([param.nelement()*param.element_size() for param in self.parameters()]))
        #print("total buf", sum([buf.nelement()*buf.element_size() for buf in self.buffers()]))

        #print("1", torch.cuda.memory_allocated("cuda"))

        if AETIME in outputlist:
            torch.cuda.synchronize()
            aestart = time.time()

        if IDENCTIME in outputlist:
            torch.cuda.synchronize()
            idencstart = time.time()

        # get identity conditioning
        if (neut_verts is None) or (neut_avgtex is None):
            raise ValueError(f'Empty identity conditioning data')

        if id_cond is None and self.id_encoder is not None:
            if (cond_avgtex is None) or (cond_verts is None):
                randomize_color = False
                if 'randomize_color' in kwargs:
                    randomize_color = kwargs.get("randomize_color")
                if self.training and randomize_color:

                    assert(False) # no randomize color

                    with torch.no_grad():
                        n = neut_avgtex.shape[0]
                        neut_avgtex_pert = []
                        for i in range(n):

                            #R, _ = cvRodrigues(np.random.randn(3) * np.pi / 6)
                            R = None

                            scale = np.random.randn(1) * 0.1 + 1.0
                            R = torch.from_numpy((scale * R).astype(np.float32)).cuda()
                            im = neut_avgtex[i] * self.texstd + self.texmean
                            im = R @ im.view(3,-1)
                            im = (im.view(3,1024,1024) - self.texmean ) / self.texstd
                            neut_avgtex_pert.append(im.unsqueeze(0))
                        neut_avgtex_pert = torch.cat(neut_avgtex_pert)

                    id_cond, idenc_losses = self.id_encoder(neut_verts, neut_avgtex_pert, losslist=losslist)
                else:
                    normalize_color = False
                    if 'normalize_color' in kwargs:
                        normalize_color = kwargs.get("normalize_color")
                    if self.training and normalize_color:
                        with torch.no_grad():
                            dst = self.texmean.unsqueeze(0)
                            neut_avgtex_pert = (color_normalize(neut_avgtex * self.texstd + dst, dst) - dst) / self.texstd

                        id_cond, idenc_losses = self.id_encoder(neut_verts, neut_avgtex_pert, losslist=losslist)
                    else:
                        id_cond, idenc_losses = self.id_encoder(neut_verts, neut_avgtex, losslist=losslist) #{z_tex_id, b_tex_id, z_geo_id, b_geo_id}

            else:
                id_cond, idenc_losses = self.id_encoder(neut_verts, neut_avgtex, cond_verts, cond_avgtex, losslist=losslist) #{z_tex_id, b_tex_id, z_geo_id, b_geo_id}

            if idenc_losses is not None:
                resultlosses.update(idenc_losses)
        else:
            idenc_losses = None


        #zero-out identity biases
        if 'zerobias' in kwargs:
            zero_out_biases = kwargs.get("zerobias")
            with torch.no_grad():
                if zero_out_biases:
                    for i in range(len(id_cond['b_geo'])):
                        id_cond['b_geo'][i] = (id_cond['b_geo'][i] * 0).detach()
                    for i in range(len(id_cond['b_tex'])):
                        id_cond['b_tex'][i] = (id_cond['b_tex'][i] * 0).detach()

        if 'id_cond' in outputlist:
            resultout['id_cond'] = id_cond

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
                    *[dict(verts=verts, avgtex=avgtex, fixedcamimage=fixedcamimage, neut_verts=neut_verts, neut_avgtex=neut_avgtex, gaze=gaze)[k] for k in self.encoderinputs],
                    losslist=losslist)

            encoding = encout["encoding"]
            #result["losses"].update(encout["losses"])
            resultlosses.update(enclosses)
            resultout['encoding'] = encoding

            if 'gaze' in encout:
                gazepred = encout['gaze']
                resultout['gazepred'] = gazepred

            if 'neckrot' in encout:
                neckrotpred = encout['neckrot']
                resultout['neckrotpred'] = neckrotpred

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
        viewdir = viewpos / torch.sqrt(torch.sum(viewpos ** 2, dim=-1, keepdim=True))

        condinput = []
        if gazepred is not None:
            condinput.append(gazepred)
        if neckrotpred is not None:
            condinput.append(neckrotpred)
        if len(condinput) > 0:
            condinput = torch.cat(condinput, dim=-1)
            decout, declosses = self.decoder(verts,
                                             id_cond,
                                             encoding, viewpos,
                                            #  condinput=condinput,
                                             renderoptions=renderoptions,
                                             trainiter=trainiter,
                                             losslist=losslist)
        else:
            # id_cond_dummy = torch.randn((4, 1))
            decout, declosses = self.decoder(verts,
                                             id_cond,
                                             encoding, viewpos,
                                            #  condinput=condinput,
                                             renderoptions=renderoptions,
                                             trainiter=trainiter,
                                             losslist=losslist)

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

            if False:#self.vertl1: #tmp
                vertsqerr = weight * torch.abs(verts - vertsrecstd)
            else:
                vertsqerr = weight * (verts - vertsrecstd) ** 2

            vertmse = torch.sum(vertsqerr.view(vertsqerr.size(0), -1), dim=-1)
            vertmse_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)

            #result["losses"]["vertmse"] = (vertmse, vertmse_weight)
            resultlosses["vertmse"] = (vertmse, vertmse_weight)

        if "vertl1" in losslist:
            weight = validinput[:, None, None].expand_as(verts)

            vertsrecstd = (decout["verts"] - self.vertmean) / self.vertstd

            vertsqerr = weight * torch.abs(verts - vertsrecstd)

            vertmse = torch.sum(vertsqerr.view(vertsqerr.size(0), -1), dim=-1)
            vertmse_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)

            resultlosses["vertl1"] = (vertmse, vertmse_weight)

        if "vertl1_mod" in losslist:
            if 'verts_mask' not in decout:
                print(f'verts_mask decoder output required for vertl1_mod loss')
                quit()
            if 'verts_gt' not in decout:
                print(f'verts_gt decoder output required for vertl1_mod loss')
                quit()
            if 'verts_pred' not in decout:
                print(f'verts_pred decoder output required for vertl1_mod loss')
                quit()
            weight = validinput[:, None, None].expand_as(decout["verts_gt"]) * decout["verts_mask"]
            vertsqerr = weight * torch.abs(decout["verts_gt"] - decout["verts_pred"])
            vertmse = torch.sum(vertsqerr.view(vertsqerr.size(0), -1), dim=-1)
            vertmse_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)
            resultlosses["vertl1_mod"] = (vertmse, vertmse_weight)

        # TODO: check this
        # subsample depth, imagerec, imagerecmask
        if image is not None and pixelcoords.size()[1:3] != image.size()[2:4]:
            samplecoords = torch.cat([pixelcoords[...,:1] * 2 / (image.shape[-1] - 1) - 1,
                                      pixelcoords[...,1:] * 2 / (image.shape[-2] - 1) - 1], dim=-1)
        else:
            samplecoords = torch.cat([pixelcoords[...,:1] * 2 / (pixelcoords.shape[-2] - 1) - 1,
                                      pixelcoords[...,1:] * 2 / (pixelcoords.shape[-3] - 1) - 1], dim=-1)

        if 'samplecoords' in outputlist:
            resultout['samplecoords'] = samplecoords

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
        raydir = raydir / torch.sqrt(torch.sum(raydir ** 2, dim=-1, keepdim=True))

        raypos, raydir, tminmax = compute_raydirs(
            campos, camrot, focal, princpt, pixelcoords, self.raymarcher.volume_radius
        )
        rayposbeg = campos[:, None, None, :] + raydir * tminmax[:, :, :, :1]
        rayposend = campos[:, None, None, :] + raydir * tminmax[:, :, :, 1:]

        rayrgb, rayalpha, rayrgba, pos_img = self.raymarcher(
            raypos, raydir, tminmax, outputs=decout, with_pos_img=('pos_img' in outputlist),
        )
        if 'pos_img' in outputlist:
            resultout['pos_img'] = pos_img

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

                    #remove chair
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

            if 'dataset_type' in kwargs:
                dataset_type = kwargs.get("dataset_type")
                bg = self.bgmodel(dataset_type, camindex, idindex, samplecoords)
            else:
                bg = self.bgmodel(camindex, idindex, samplecoords)

        if 'bg' in outputlist:
            resultout['bg'] = bg

        # 6. matting
        rastrgb = None
        bgcolor = None
        if rastrgb is not None:
            if bg is not None:
                rayrgb = rayrgb + (1. - rayalpha) * (rlirgbrec + (1. - rlialpharec) * bg)
            else:
                rayrgb = rayrgb + (1. - rayalpha) * rlirgbrec
        else:
            if bgcolor is not None:
                rayrgb = rayrgb + (1. - rayalpha) * bgcolor.to('cuda')[None,:,None,None]
            elif bg is not None:
                rayrgb = rayrgb + (1. - rayalpha) * bg
            else:
                c = np.asarray([0,0,0], dtype=np.float32)
                rayrgb = rayrgb + (1. - rayalpha) * torch.from_numpy(c).to('cuda')[None,:,None,None]


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
                #resultout["sampledimg_smooth"] = image_smooth

            # img = einops.rearrange(image, "n c h w -> n h w c")
            # im = Image.fromarray(img[0].cpu().numpy().astype(np.uint8))
            # im.save("/checkpoint/avatar/julietamartinez/oss_release/meow2.png")

            # img = einops.rearrange(rayrgb, "n c h w -> n h w c")
            # im = Image.fromarray(img[0].detach().cpu().numpy().astype(np.uint8))
            # im.save("/checkpoint/avatar/julietamartinez/oss_release/meow3.png")

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

            if 'imageweight' in outputlist:
                resultout['imageweight'] = weight

            irgbsqerr = (weight * (image - rayrgb) ** 2).contiguous()

            if "irgbsqerr" in outputlist:
                resultout["irgbsqerr"] = irgbsqerr

            if "irgbmse" in losslist:
                irgbabs = (weight * ((image - rayrgb)**2)).contiguous()

                irgbl2 = torch.sum(irgbabs.view(irgbabs.size(0), -1), dim=-1)
                irgbl2_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)

                resultlosses["irgbmse"] = (irgbl2, irgbl2_weight.clamp(min=1e-6))

            if "irgbl1" in losslist:
                irgbabs = (weight * torch.abs(image - rayrgb)).contiguous()

                irgbl1 = torch.sum(irgbabs.view(irgbabs.size(0), -1), dim=-1)
                irgbl1_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)

                resultlosses["irgbl1"] = (irgbl1, irgbl1_weight.clamp(min=1e-6))

            if "vgg" in losslist:
                if self.net_vgg is None:
                    print('VGG model not set for VGGLoss computation!')
                    quit()
                x1 = weight * image
                #x1 = weight * image_smooth
                x2 = weight * rayrgb
                if x1.shape[-2] > 640:
                    h = 640
                    w = int((h / x1.shape[-2]) * x1.shape[-1])
                    x1 = F.interpolate(x1, size=(h, w), mode='bilinear')
                    x2 = F.interpolate(x2, size=(h, w), mode='bilinear')
                x1 = self.net_vgg.calc_feats(x1)
                x2 = self.net_vgg.calc_feats(x2)
                vgg_loss = self.net_vgg(x1, x2, None)
                resultlosses["vgg"] = vgg_loss

            if "ialphal1" in losslist:
                if segmentation is None:
                    print('Asked for alphal1 loss but no segmentation found')
                    quit()
                with torch.no_grad():
                    alpha = ((torch.abs(segmentation - 1) < 0.5) | (torch.abs(segmentation - 2) < 0.5) | (torch.abs(segmentation - 4) < 0.5)).float()

                segmask = (torch.sum(segmentation.view(segmentation.shape[0], -1), dim=-1) > 0).float()

                ialphaabs = (chair * torch.abs(alpha - rayalpha)).contiguous() * segmask[:,None,None,None]
                ialphal1 = torch.sum(ialphaabs.view(ialphaabs.size(0), -1), dim=-1)
                ialphal1_weight = torch.sum(chair.view(chair.size(0), -1), dim=-1)
                resultlosses["ialphal1"] = (ialphal1, ialphal1_weight.clamp(min=1e-6))

            if "ialphabce" in losslist:
                if segmentation is None:
                    print('Asked for alphal1 loss but no segmentation found')
                    quit()
                with torch.no_grad():
                    alpha = ((torch.abs(segmentation - 1) < 0.5) | (torch.abs(segmentation - 2) < 0.5) | (torch.abs(segmentation - 4) < 0.5)).float()

                L = torch.nn.BCELoss(reduction='none')
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



    def encode_identity(self,
                        neut_avgtex=None,
                        neut_verts=None,
                        cond_avgtex=None,
                        cond_verts=None,
                        **kwargs):

        # get identity conditioning
        if (neut_verts is None) or (neut_avgtex is None):
            print(f'Empty identity conditioning data')
            return None

        if (cond_avgtex is None) or (cond_verts is None):
            randomize_color = False
            if 'randomize_color' in kwargs:
                randomize_color = kwargs.get("randomize_color")
            if self.training and randomize_color:
                with torch.no_grad():
                    n = neut_avgtex.shape[0]
                    neut_avgtex_pert = []
                    for i in range(n):
                        #R, _ = cvRodrigues(np.random.randn(3) * np.pi / 6)
                        R = None
                        scale = np.random.randn(1) * 0.1 + 1.0
                        R = torch.from_numpy((scale * R).astype(np.float32)).cuda()
                        im = neut_avgtex[i] * self.texstd + self.texmean
                        im = R @ im.view(3,-1)
                        im = (im.view(3,1024,1024) - self.texmean ) / self.texstd
                        neut_avgtex_pert.append(im.unsqueeze(0))
                    neut_avgtex_pert = torch.cat(neut_avgtex_pert)

                id_cond, _ = self.id_encoder(neut_verts, neut_avgtex_pert, losslist=[])
            else:
                normalize_color = False
                if 'normalize_color' in kwargs:
                    normalize_color = kwargs.get("normalize_color")
                if self.training and normalize_color:
                    with torch.no_grad():
                        dst = self.texmean.unsqueeze(0)
                        neut_avgtex_pert = (color_normalize(neut_avgtex * self.texstd + dst, dst) - dst) / self.texstd

                    id_cond, _ = self.id_encoder(neut_verts, neut_avgtex_pert, losslist=[])
                else:
                    id_cond, _ = self.id_encoder(neut_verts, neut_avgtex, losslist=[]) #{z_tex_id, b_tex_id, z_geo_id, b_geo_id}

        else:
            id_cond, _ = self.id_encoder(neut_verts, neut_avgtex, cond_verts, cond_avgtex, losslist=[]) #{z_tex_id, b_tex_id, z_geo_id, b_geo_id}

        #zero-out identity biases
        if 'zerobias' in kwargs:
            zero_out_biases = kwargs.get("zerobias")
            with torch.no_grad():
                if zero_out_biases:
                    for i in range(len(id_cond['b_geo'])):
                        id_cond['b_geo'][i] = (id_cond['b_geo'][i] * 0).detach()
                    for i in range(len(id_cond['b_tex'])):
                        id_cond['b_tex'][i] = (id_cond['b_tex'][i] * 0).detach()

        return id_cond



    def encode_expression(self,
                          encoder = None,
                          relheadrot : Optional[torch.Tensor] = None,
                          gaze : Optional[torch.Tensor] = None,
                          avgtex : Optional[torch.Tensor]=None,
                          verts : Optional[torch.Tensor]=None,
                          neut_avgtex : Optional[torch.Tensor]=None,
                          neut_verts : Optional[torch.Tensor]=None,
                          cond_avgtex : Optional[torch.Tensor]=None,
                          cond_verts : Optional[torch.Tensor]=None,
                          fixedcamimage : Optional[torch.Tensor]=None,
                          losslist : list=[],
                          **kwargs):


        resultout, resultlosses = {}, {}

        if encoder is None:
            encout, enclosses = self.encoder(
                *[dict(verts=verts, avgtex=avgtex, fixedcamimage=fixedcamimage, neut_verts=neut_verts, neut_avgtex=neut_avgtex, gaze=gaze)[k] for k in self.encoderinputs],
                losslist=losslist)
        else:
            encout, enclosses = encoder(
                *[dict(verts=verts, avgtex=avgtex, fixedcamimage=fixedcamimage, neut_verts=neut_verts, neut_avgtex=neut_avgtex, gaze=gaze)[k] for k in self.encoderinputs],
                losslist=losslist)

        encoding = encout["encoding"]
        resultlosses.update(enclosses)
        resultout['encoding'] = encoding

        if 'gaze' in encout:
            gazepred = encout['gaze']
            resultout['gazepred'] = gazepred

        if 'neckrot' in encout:
            neckrotpred = encout['neckrot']
            resultout['neckrotpred'] = neckrotpred

        return resultout, resultlosses




    #@profile
    def decode(self,
               verts,
               camrot,
               campos,
               focal,
               princpt,
               id_cond,
               encoding,
               pixelcoords,
               bg = None,
               bgcolor = None,
               relheadrot = None,
               gaze = None,
               neckrot = None,
               idindex = None,
               camindex = None,
               modelmatrix = None,
               renderoptions = {},
               trainiter = -1,
               outputlist = [],
               **kwargs):

        resultout = {}

        # compute relative viewing position
        viewrot = torch.bmm(camrot, modelmatrix[:, :3, :3])
        viewpos = torch.bmm((campos[:, :] - modelmatrix[:, :3, 3])[:, None, :], modelmatrix[:, :3, :3])[:, 0, :]
        viewdir = viewpos / torch.sqrt(torch.sum(viewpos ** 2, dim=-1, keepdim=True))

        condinput = []
        if gaze is not None:
            condinput.append(gaze)
        if neckrot is not None:
            condinput.append(neckrot)
        if len(condinput) > 0:
            condinput = torch.cat(condinput, dim=-1)
            decout, declosses = self.decoder(verts,
                                             id_cond, encoding, viewpos,
                                             condinput=condinput,
                                             renderoptions=renderoptions,
                                             trainiter=trainiter,
                                             losslist=[])
        else:
            decout, declosses = self.decoder(verts,
                                             id_cond, encoding, viewpos,
                                             renderoptions=renderoptions,
                                             trainiter=trainiter,
                                             losslist=[])

        # NHWC
        raydir = (pixelcoords - princpt[:, None, None, :]) / focal[:, None, None, :]
        raydir = torch.cat([raydir, torch.ones_like(raydir[:, :, :, 0:1])], dim=-1)
        raydir = torch.sum(viewrot[:, None, None, :, :] * raydir[:, :, :, :, None], dim=-2)
        raydir = raydir / torch.sqrt(torch.sum(raydir ** 2, dim=-1, keepdim=True))

        raypos, raydir, tminmax = compute_raydirs(
            campos, camrot, focal, princpt, pixelcoords, self.raymarcher.volume_radius
        )
        rayrgb, rayalpha, rayrgba, pos_img = self.raymarcher(
            raypos, raydir, tminmax, outputs=decout, with_pos_img=('pos_img' in outputlist),#self.with_depth
        )
        if 'pos_img' in outputlist:
            resultout['pos_img'] = pos_img

        # color correction
        if (camindex is not None) and (idindex is not None):
            rayrgb = self.colorcal(rayrgb, camindex, idindex)

        if bg is not None:
            rayrgb = rayrgb + (1. - rayalpha) * bg

        if "irgbrec" in outputlist:
            resultout["irgbrec"] = rayrgb

        if "ialpha" in outputlist:
            resultout["ialpha"] = rayalpha

        return resultout



################################################################################
from torchvision import models as std_models #only for vgg
import os
#os.environ['TORCH_HOME'] = '/mnt/home/jsaragih/models/'

os.environ['TORCH_HOME'] = '/checkpoint/avatar/assets/models'

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = std_models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLossMasked2(nn.Module):
    def __init__(self, weights = None):
        super(VGGLossMasked2, self).__init__()
        self.vgg = Vgg19()
        if weights is None:
            self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        else:
            self.weights = weights

    def normalize_batch(self, batch):
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (batch / 255. - mean) / std

    def calc_feats(self, x):
        return self.vgg(self.normalize_batch(x))

    def gram(self, feats, mask = None):
        if mask is not None:
            m = F.interpolate(mask, size=(feats.shape[-2], feats.shape[-1]),\
                              mode='bilinear').detach()

            f = (feats * m).view(feats.shape[0], feats.shape[1],-1)

            G = torch.bmm(f, f.permute(0,2,1)) / f.shape[-1] #(torch.sum(m) + 1e-8)

            #G = torch.sum(f[:,:,None,:] * f[:,None,:,:], dim=-1, keepdims=False) /\
            #(torch.sum(m) + 1e-8)
        else:
            f = feats.view(feats.shape[0], feats.shape[1],-1)
            G = torch.bmm(f, f.permute(0,2,1)) / f.shape[-1]
            #G = torch.mean(f[:,:,None,:] * f[:,None,:,:], dim=-1, keepdims=False)
        return G

    def forward(self, x_vgg, y_vgg, mask):

        loss = 0
        for i in range(len(x_vgg)):
            if mask is not None:
                m = F.interpolate(mask, size=(x_vgg[i].shape[-2],\
                                              x_vgg[i].shape[-1]),\
                                  mode='bilinear').detach()
                vx = x_vgg[i] * m
                vy = y_vgg[i] * m
                dd = vx.shape[1]
                loss = loss + self.weights[i] * torch.sum((vx - vy).abs()) / dd
            else:
                #dd = x_vgg[i].shape[1]
                #loss = loss + self.weights[i] * \
                #  torch.sum((x_vgg[i] - y_vgg[i]).abs()) / dd
                loss = loss + self.weights[i] * \
                       torch.mean((x_vgg[i] - y_vgg[i]).abs())
        return loss





############################################

from typing import Tuple


class GaussianBlur2d(nn.Module):
    r"""Creates an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.

    Arguments:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> gauss = GaussianBlur2d((3, 3), (1.5, 1.5))
        >>> output = gauss(input)  # 2x4x5x5
    """

    def __init__(self, kernel_size: Tuple[int, int],
                 sigma: Tuple[float, float],
                 border_type: str = 'reflect') -> None:
        super(GaussianBlur2d, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self.kernel: torch.Tensor = torch.unsqueeze(
            get_gaussian_kernel2d(kernel_size, sigma), dim=0)

        assert border_type in ["constant", "reflect", "replicate", "circular"]
        self.border_type = border_type

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(kernel_size=' + str(self.kernel_size) + ', ' +\
            'sigma=' + str(self.sigma) + ', ' +\
            'border_type=' + self.border_type + ')'

    def forward(self, x: torch.Tensor):  # type: ignore
        return filter2D(x, self.kernel, self.border_type)

def get_gaussian_kernel2d(
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])

        >>> get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(
                kernel_size
            )
        )
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma)
        )
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t()
    )
    return kernel_2d

def get_gaussian_kernel1d(kernel_size: int,
                          sigma: float,
                          force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples::

        >>> get_gaussian_kernel(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> get_gaussian_kernel(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d

def gaussian(window_size, sigma):
    x = torch.arange(window_size).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / float(2 * sigma ** 2)))
    return gauss / gauss.sum()

#######################################
# kornia/filter/filter.py related code.
######################################

def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalizes both derivative and smoothing kernel.
    """
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))

def compute_padding(kernel_size: Tuple[int, int]):
    """Computes padding tuple."""
    # 4 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pyth.org/docs/stable/nn.html#th.nn.functional.pad
    assert len(kernel_size) == 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymetric padding :(
    return [computed[1] - 1 if kernel_size[0] % 2 == 0 else computed[1],
            computed[1],
            computed[0] - 1 if kernel_size[1] % 2 == 0 else computed[0],
            computed[0]]


def filter2D(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect',
             normalized: bool = False) -> torch.Tensor:
    r"""Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input kernel type is not a torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3:
        raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         .format(kernel.shape))

    borders_list: List[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError("Invalid border_type, we expect the following: {0}."
                         "Got: {1}".format(borders_list, border_type))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(0).to(input.device).to(input.dtype)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)
    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding((height, width))
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)
    b, c, hp, wp = input_pad.shape
    # convolve the tensor with the kernel. Pick the fastest alg
    kernel_numel: int = height * width
    if kernel_numel > 81:
        return F.conv2d(input_pad.reshape(b * c, 1, hp, wp), tmp_kernel, padding=0, stride=1).view(b, c, h, w)
    return F.conv2d(input_pad, tmp_kernel.expand(c, -1, -1, -1), groups=c, padding=0, stride=1)




############################################
