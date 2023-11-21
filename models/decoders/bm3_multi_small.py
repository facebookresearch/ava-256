import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import models.utils

from torch.func import vmap, stack_module_state, functional_call
import time

from math import sqrt


class MetaDecoder(nn.Module):

    def __init__(self, decoders, in_dims):
        """Create n decoders and dispatch them dynamically"""
        super().__init__()

        params, buffers = stack_module_state(decoders)

        # ParameterDict doesn't allow keys wtih `.` so we need to replace it with `__`
        assert False == any("__" in k for k in params.keys())
        self.decoders_params = nn.ParameterDict({k.replace('.', '__'): v for k, v in params.items()})

        self.decoder_buffer_keys = list(buffers.keys())
        for k in self.decoder_buffer_keys:
            self.register_buffer(k, buffers[k])

        self.register_buffer("num_iters", torch.zeros(len(decoders), dtype=torch.int32))

        base_model = copy.deepcopy(decoders[0])
        base_model = base_model.to("meta")
        def fmodel(params, buffers, x):
            return functional_call(base_model, (params, buffers), x)
        self.func = fmodel
        self.in_dims = in_dims


    def forward(self, i, x):
        """
        Args:
            i (int): the index of the decoder to use
            x (tuple): everything that the forward function of the decoder takes
        """

        # Keep track of how many iterations we have done
        self.num_iters[i] += 1

        # retrive the parameters for decoders of corresponding index
        params = {k.replace('__', '.'): v[i] for k, v in self.decoders_params.items()}
        buffers = {k: getattr(self, k)[i] for k in self.decoder_buffer_keys}

        batched_fn = vmap(
            self.func,
            in_dims=self.in_dims,
            chunk_size=len(i),
            out_dims=1,
        )

        return batched_fn(params, buffers, x)


class Decoder5NoBias(nn.Module):
    def __init__(self, vt, vi, vti, vertmean, vertstd, volradius, #dectype="deconv",
                 nprims=128*128, primsize=(8,8,8),
                 alphafade=False, postrainstart=0, condsize=0,
                 motiontype="deconv",
                 enable_id_encoder=True, cfg=None, n_decoders=1,
                 **kwargs):
        super(Decoder5NoBias, self).__init__()

        self.cfg = cfg
        self.volradius = volradius
        self.alphafade = alphafade
        self.postrainstart = postrainstart

        self.nprims = nprims
        self.primsize = primsize
        self.motiontype = motiontype
        self.enable_id_encoder = enable_id_encoder


        self.rodrig = models.utils.Rodrigues()

        imsize = int(sqrt(nprims)) * primsize[1]
        # self.rgbdec = DecoderSlab(imsize, nprims, primsize[0], 3, viewcond=True, texwarp=False, enable_id_encoder=enable_id_encoder, cfg=cfg)
        # self.geodec = DecoderGeoSlab2(vt, vi, vti, vertmean.shape[-2], {256: 16, 16384: 128}[nprims], 256, imsize, nprims, primsize[0], enable_id_encoder=enable_id_encoder, cfg=cfg)

        rgbdecs = [DecoderSlab(imsize, nprims, primsize[0], 3, viewcond=True, texwarp=False, enable_id_encoder=enable_id_encoder, cfg=cfg) for i in range(n_decoders)]
        self.rgbdec = MetaDecoder(rgbdecs, in_dims=(0, 0, (0, None, None, 0, None, None)) )

        geodecs = [DecoderGeoSlab2(vt, vi, vti, vertmean.shape[-2], {256: 16, 16384: 128}[nprims], 256, imsize, nprims, primsize[0], enable_id_encoder=enable_id_encoder, cfg=cfg) for i in range(n_decoders)]
        self.geodec = MetaDecoder(geodecs, in_dims=(0, 0, (0, None, None, None)) )

        self.warpdec = None

        self.register_buffer("vt", torch.tensor(vt), persistent=False)
        self.register_buffer("vertmean", vertmean, persistent=False)
        self.vertstd = vertstd

        # apath=os.getenv('RSC_AVATAR_RSCASSET_PATH')
        apath = "/checkpoint/avatar/jinkyuk/rsc-assets"
        idximpath = f"{apath}/idxmap"

        self.register_buffer("idxim",
                torch.tensor(np.load("{}/retop_idxim_1024.npy".format(idximpath))).long(), persistent=False)
        self.register_buffer("tidxim",
                torch.tensor(np.load("{}/retop_tidxim_1024.npy".format(idximpath))).long(), persistent=False)
        self.register_buffer("barim",
                torch.tensor(np.load("{}/retop_barim_1024.npy".format(idximpath))), persistent=False)

        self.register_buffer("adaptwarps", 0*torch.ones(self.nprims))

    def forward(self,
                idindex,  # <-- to index into decoders
                gt_geo,
                id_cond,
                encoding,
                viewpos,
                # condinput=None,
                renderoptions={},
                trainiter=-1,
                losslist=[]):
        """
        Parameters
        ----------
        encoding : torch.Tensor [B, 256]
            Encoding of current frame
        viewpos : torch.Tensor [B, 3]
            Viewing position of target camera view
        condinput : torch.Tensor [B, ?]
            Additional conditioning input (e.g., headpose)
        renderoptions : dict
            Options for rendering (e.g., rendering debug images)
        trainiter : int,
            Current training iteration
        losslist : list,
            List of losses to compute and return
        Returns
        -------
        result : dict,
            Contains predicted vertex positions, primitive contents and
            locations, scaling, and orientation, and any losses.
        """
        nprims = self.nprims
        bsize = encoding.shape[0]

        # if condinput is not None:
        #     encoding = torch.cat([encoding, condinput], dim=1)

        z_geo, b_geo = (id_cond['z_geo'], id_cond['b_geo']) if self.enable_id_encoder else (None, None)
        primalpha, geo, primposresid, primrvecresid, primscaleresid = self.geodec(idindex, (encoding, z_geo, b_geo, trainiter))
        primalpha = primalpha.squeeze(dim=0)
        geo = geo.squeeze(dim=0)
        primposresid = primposresid.squeeze(dim=0)
        primrvecresid = primrvecresid.squeeze(dim=0)
        primscaleresid = primscaleresid.squeeze(dim=0)

        geo = geo * self.vertstd + self.vertmean

        ##########################################################################################################
        geo_orig = geo
        if trainiter <= self.postrainstart:
            geo = gt_geo * self.vertstd + self.vertmean
        ##########################################################################################################

        postex = (
            self.barim[:, :, 0, None] * geo.index_select(1, self.idxim[:, :, 0].reshape(-1)).reshape(-1, 1024, 1024, 3) +
            self.barim[:, :, 1, None] * geo.index_select(1, self.idxim[:, :, 1].reshape(-1)).reshape(-1, 1024, 1024, 3) +
            self.barim[:, :, 2, None] * geo.index_select(1, self.idxim[:, :, 2].reshape(-1)).reshape(-1, 1024, 1024, 3)
        ).permute(0, 3, 1, 2) / self.volradius

        # placement of primitives on mesh
        if self.nprims == 1:
            primpos = torch.zeros((postex.size(0), 1, 3), device=postex.device)
            primscale = 2.

            v0 = torch.tensor([0., 0., 0.], device=postex.device)[None, None, None, :].repeat(postex.size(0), 1, 1, 1)
            v1 = torch.tensor([1., 0., 0.], device=postex.device)[None, None, None, :].repeat(postex.size(0), 1, 1, 1)
            v2 = torch.tensor([0., 1., 0.], device=postex.device)[None, None, None, :].repeat(postex.size(0), 1, 1, 1)
        elif self.nprims == 8:
            primpos = postex[:, :, 256::512, 128::256].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 4.

            v0 = geo[:, self.idxim[256::512, 128::256, 0], :] # vert 0?
            v1 = geo[:, self.idxim[256::512, 128::256, 1], :] # vert 1?
            v2 = geo[:, self.idxim[256::512, 128::256, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[256::512, 128::256, 0], :]
            vt1 = self.vt[self.tidxim[256::512, 128::256, 1], :]
            vt2 = self.vt[self.tidxim[256::512, 128::256, 2], :]
        elif self.nprims == 64:
            primpos = postex[:, :, 64::128, 64::128].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 8.

            v0 = geo[:, self.idxim[64::128, 64::128, 0], :] # vert 0?
            v1 = geo[:, self.idxim[64::128, 64::128, 1], :] # vert 1?
            v2 = geo[:, self.idxim[64::128, 64::128, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[64::128, 64::128, 0], :]
            vt1 = self.vt[self.tidxim[64::128, 64::128, 1], :]
            vt2 = self.vt[self.tidxim[64::128, 64::128, 2], :]
        elif self.nprims == 256:
            primpos = postex[:, :, 32::64, 32::64].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 12.

            ###################################################
            if trainiter < 100:
                with torch.no_grad():
                    centdiffx = postex[:, :, 32::64, (32+64)::64] - postex[:, :, 32::64, 32:-64:64]
                    centdiffx = torch.cat((centdiffx, centdiffx[:,:,:,-1:]), dim=3)
                    centdiffy = postex[:, :, (32+64)::64, 32::64] - postex[:, :, 32:-64:64, 32::64]
                    centdiffy = torch.cat((centdiffy, centdiffy[:,:,-1:,:]), dim=2)
                    centdiffx = centdiffx.norm(dim=1)
                    centdiffy = centdiffy.norm(dim=1)
                    centsize = torch.max(centdiffx,centdiffy)
                    centsize = torch.max(centsize,dim=0)[0].view(self.nprims)
                    warps_vec = 2/centsize
                    if self.adaptwarps.max().item()==0:
                        self.adaptwarps.data[:] = warps_vec
                    else:
                        self.adaptwarps.data[:] = self.adaptwarps*0.9 + 0.1*warps_vec
            primscale = self.adaptwarps[None,:,None] * 0.8
            ###################################################

            v0 = geo[:, self.idxim[32::64, 32::64, 0], :] # vert 0?
            v1 = geo[:, self.idxim[32::64, 32::64, 1], :] # vert 1?
            v2 = geo[:, self.idxim[32::64, 32::64, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[32::64, 32::64, 0], :]
            vt1 = self.vt[self.tidxim[32::64, 32::64, 1], :]
            vt2 = self.vt[self.tidxim[32::64, 32::64, 2], :]

            geodu = postex[:, :, :, 1:] - postex[:, :, :, :-1]
            geodv = postex[:, :, 1:, :] - postex[:, :, :-1, :]
            vcenterdu = geodu[:, :, 32::64, 32::64].permute(0,2,3,1)
            vcenterdv = geodv[:, :, 32::64, 32::64].permute(0,2,3,1) # match v0 (channels last)

        elif self.nprims == 512:
            primpos = postex[:, :, 32::64, 16::32].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 16.

            v0 = geo[:, self.idxim[32::64, 16::32, 0], :] # vert 0?
            v1 = geo[:, self.idxim[32::64, 16::32, 1], :] # vert 1?
            v2 = geo[:, self.idxim[32::64, 16::32, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[32::64, 16::32, 0], :]
            vt1 = self.vt[self.tidxim[32::64, 16::32, 1], :]
            vt2 = self.vt[self.tidxim[32::64, 16::32, 2], :]
        elif self.nprims == 4096:
            primpos = postex[:, :, 8::16, 8::16].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 32.

            v0 = geo[:, self.idxim[8::16, 8::16, 0], :] # vert 0?
            v1 = geo[:, self.idxim[8::16, 8::16, 1], :] # vert 1?
            v2 = geo[:, self.idxim[8::16, 8::16, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[8::16, 8::16, 0], :]
            vt1 = self.vt[self.tidxim[8::16, 8::16, 1], :]
            vt2 = self.vt[self.tidxim[8::16, 8::16, 2], :]
        elif self.nprims == 16384:

            primpos = postex[:, :, 4::8, 4::8].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 48.

            ###################################################
            if trainiter < 100:
                with torch.no_grad():
                    centdiffx = postex[:, :, 4::8, (4+8)::8] - postex[:, :, 4::8, 4:-8:8]
                    centdiffx = torch.cat((centdiffx, centdiffx[:,:,:,-1:]), dim=3)
                    centdiffy = postex[:, :, (4+8)::8, 4::8] - postex[:, :, 4:-8:8, 4::8]
                    centdiffy = torch.cat((centdiffy, centdiffy[:,:,-1:,:]), dim=2)
                    centdiffx = centdiffx.norm(dim=1)
                    centdiffy = centdiffy.norm(dim=1)
                    centsize = torch.max(centdiffx,centdiffy)
                    centsize = torch.max(centsize,dim=0)[0].view(self.nprims)
                    warps_vec = 2/centsize
                    if self.adaptwarps.max().item()==0:
                        self.adaptwarps.data[:] = warps_vec
                    else:
                        self.adaptwarps.data[:] = self.adaptwarps*0.9 + 0.1*warps_vec
            primscale = self.adaptwarps[None,:,None] * 0.8
            ###################################################

            v0 = geo[:, self.idxim[4::8, 4::8, 0], :] # vert 0?
            v1 = geo[:, self.idxim[4::8, 4::8, 1], :] # vert 1?
            v2 = geo[:, self.idxim[4::8, 4::8, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[4::8, 4::8, 0], :]
            vt1 = self.vt[self.tidxim[4::8, 4::8, 1], :]
            vt2 = self.vt[self.tidxim[4::8, 4::8, 2], :]

            geodu = postex[:, :, :, 1:] - postex[:, :, :, :-1]
            geodv = postex[:, :, 1:, :] - postex[:, :, :-1, :]
            vcenterdu = geodu[:, :, 4::8, 4::8].permute(0,2,3,1)
            vcenterdv = geodv[:, :, 4::8, 4::8].permute(0,2,3,1) # match v0 (channels last)

        elif self.nprims == 32768:
            primpos = postex[:, :, 4::8, 2::4].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 64.

            v0 = geo[:, self.idxim[4::8, 2::4, 0], :] # vert 0?
            v1 = geo[:, self.idxim[4::8, 2::4, 1], :] # vert 1?
            v2 = geo[:, self.idxim[4::8, 2::4, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[4::8, 2::4, 0], :]
            vt1 = self.vt[self.tidxim[4::8, 2::4, 1], :]
            vt2 = self.vt[self.tidxim[4::8, 2::4, 2], :]
        elif self.nprims == 262144:
            primpos = postex[:, :, 1::2, 1::2].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 128.

            v0 = geo[:, self.idxim[1::2, 1::2, 0], :] # vert 0?
            v1 = geo[:, self.idxim[1::2, 1::2, 1], :] # vert 1?
            v2 = geo[:, self.idxim[1::2, 1::2, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[1::2, 1::2, 0], :]
            vt1 = self.vt[self.tidxim[1::2, 1::2, 1], :]
            vt2 = self.vt[self.tidxim[1::2, 1::2, 2], :]
        else:
            raise


        tangent = vcenterdu
        tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-8)
        normal = torch.cross(tangent, vcenterdv, dim=3)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True).clamp(min=1e-8)
        bitangent = torch.cross(normal, tangent, dim=3)
        bitangent = bitangent / torch.norm(bitangent, dim=-1, keepdim=True).clamp(min=1e-8)
        primrot = torch.stack((tangent, bitangent, normal), dim=-2).view(encoding.size(0), -1, 3, 3).contiguous().permute(0, 1, 3, 2).contiguous()

        #primposresid, primrvecresid, primscaleresid = self.motiondec(encoding)
        if trainiter <= self.postrainstart:
        #if True: #########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            primposresid = primposresid * 0.
            primrvecresid = primrvecresid * 0.
            primscaleresid = primscaleresid * 0. + 1.
        elif trainiter <= 2 * self.postrainstart:
            weight = (2 * self.postrainstart - trainiter) / (self.postrainstart + 1)
            primposresid = primposresid * (1-weight) + weight * 0.
            primrvecresid = primrvecresid * (1-weight) + weight * 0.
            primscaleresid = primscaleresid * (1-weight) + weight * 1.

        primrotresid = self.rodrig(primrvecresid.view(-1, 3)).view(encoding.size(0), nprims, 3, 3)

        primpos = primpos + primposresid
        primrot = torch.bmm(primrot.view(-1, 3, 3), primrotresid.view(-1, 3, 3)).view(encoding.size(0), nprims, 3, 3)
        primscale = primscale * primscaleresid

        viewdirs = viewpos / torch.sqrt(torch.sum(viewpos ** 2, dim=1, keepdim=True))

        z_tex, b_tex = (id_cond['z_tex'], id_cond['b_tex']) if self.enable_id_encoder else (None, None)

        use_warp = False
        iternum = trainiter
        primrgb = self.rgbdec(idindex, (encoding, z_tex, b_tex, viewdirs, use_warp, iternum))
        primrgb = primrgb.squeeze(dim=0)

        template = torch.cat([F.relu(primrgb * 25. + 100.), F.relu(primalpha)], dim=2)

        if self.warpdec is not None:
            warp = self.warpdec(encoding, trainiter=trainiter) * 0.01 + torch.stack(torch.meshgrid(
                torch.linspace(-1., 1., self.primsize[0], device=encoding.device),
                torch.linspace(-1., 1., self.primsize[0], device=encoding.device),
                torch.linspace(-1., 1., self.primsize[0], device=encoding.device))[::-1], dim=0)[None, None, :, :, :, :]
        else:
            warp = None

        if self.alphafade:
            gridz, gridy, gridx = torch.meshgrid(
                    torch.linspace(-1., 1., self.primsize, device=encoding.device),
                    torch.linspace(-1., 1., self.primsize, device=encoding.device),
                    torch.linspace(-1., 1., self.primsize, device=encoding.device))
            grid = torch.stack([gridx, gridy, gridz], dim=-1)
            template = template * torch.stack([
                gridx * 0. + 1.,
                gridy * 0. + 1.,
                gridz * 0. + 1.,
                torch.exp(-8.0 * torch.sum(grid ** 8, dim=-1))], dim=0)[None, None, :, :, :, :]

        if "viewaxes" in renderoptions and renderoptions["viewaxes"]:
            template[:, :, 3, template.size(3)//2:template.size(3)//2+1, template.size(4)//2:template.size(4)//2+1, :] = 2550.
            template[:, :, 0, template.size(3)//2:template.size(3)//2+1, template.size(4)//2:template.size(4)//2+1, :] = 2550.
            template[:, :, 3, template.size(3)//2:template.size(3)//2+1, :, template.size(5)//2:template.size(5)//2+1] = 2550.
            template[:, :, 1, template.size(3)//2:template.size(3)//2+1, :, template.size(5)//2:template.size(5)//2+1] = 2550.
            template[:, :, 3, :, template.size(4)//2:template.size(4)//2+1, template.size(5)//2:template.size(5)//2+1] = 2550.
            template[:, :, 2, :, template.size(4)//2:template.size(4)//2+1, template.size(5)//2:template.size(5)//2+1] = 2550.

        if "colorprims" in renderoptions and renderoptions["colorprims"]:
            lightdir = torch.tensor([1., 1., 1.], device=template.device)
            lightdir = lightdir / torch.sqrt(torch.sum(lightdir ** 2))
            zz, yy, xx = torch.meshgrid(
                torch.linspace(-1., 1., template.size(-1), device=template.device),
                torch.linspace(-1., 1., template.size(-1), device=template.device),
                torch.linspace(-1., 1., template.size(-1), device=template.device))
            primnormalx = torch.where(
                    (torch.abs(xx) >= torch.abs(yy)) & (torch.abs(xx) >= torch.abs(zz)),
                    torch.sign(xx) * torch.ones_like(xx),
                    torch.zeros_like(xx))
            primnormaly = torch.where(
                    (torch.abs(yy) >= torch.abs(xx)) & (torch.abs(yy) >= torch.abs(zz)),
                    torch.sign(yy) * torch.ones_like(xx),
                    torch.zeros_like(xx))
            primnormalz = torch.where(
                    (torch.abs(zz) >= torch.abs(xx)) & (torch.abs(zz) >= torch.abs(yy)),
                    torch.sign(zz) * torch.ones_like(xx),
                    torch.zeros_like(xx))
            primnormal = torch.stack([primnormalx, primnormaly, primnormalz], dim=-1)
            primnormal = primnormal / torch.sqrt(torch.sum(primnormal ** 2, dim=-1, keepdim=True))
            template[:, :, 3, :, :, :] = 1000.
            np.random.seed(123456)
            for i in range(template.size(1)):
                template[:, i, 0, :, :, :] = np.random.rand() * 255.
                template[:, i, 1, :, :, :] = np.random.rand() * 255.
                template[:, i, 2, :, :, :] = np.random.rand() * 255.

                lightdir0 = torch.sum(primrot[:, i, :, :] * lightdir[None, :, None], dim=-2)
                template[:, i, :3, :, :, :] *= 1.2 * torch.sum(
                        lightdir0[:, None, None, None, :] * primnormal, dim=-1)[:, None, :, :, :].clamp(min=0.05)

        # visualize slab
        if "viewslab" in renderoptions and renderoptions["viewslab"]:
            yy, xx = torch.meshgrid(
                    torch.linspace(0.9, -0.9, 16, device=template.device),
                    torch.linspace(-0.9, 0.9, 16, device=template.device))
            primpos = torch.stack([xx, yy, xx*0.], dim=-1)[None, :, :, :].repeat(template.size(0), 1, 1, 1).view(-1, 256, 3)
            primrot = torch.eye(3, device=template.device)[None, None, :, :].repeat(template.size(0), 256, 1, 1)
            primscale = torch.ones((template.size(0), 256, 3), device=template.device) * 16.

        losses = {}

        if "primvolsum" in losslist:
            losses["primvolsum"] = torch.sum(torch.prod(1. / primscale, dim=-1), dim=-1)

        #######################################################################################################
        if trainiter <= self.postrainstart:
            geo = geo_orig
        #######################################################################################################


        return {
                "verts": geo,
                "template": template,
                # "warp": warp,
                "primpos": primpos,
                "primrot": primrot,
                "primscale": primscale}, losses


###############################################################################
class DecoderSlab(nn.Module):
    def __init__(self, imsize, nboxes, boxsize, outch, viewcond=False, texwarp=False, enable_id_encoder=True, cfg=None):
        super(DecoderSlab, self).__init__()

        self.cfg = cfg
        self.imsize = imsize
        self.nboxes = nboxes
        self.boxsize = boxsize
        self.outch = outch
        self.texwarp = texwarp
        self.viewcond = viewcond


        nh = int(np.sqrt(self.nboxes))
        assert nh*nh==self.nboxes
        if nh==512:
            assert boxsize==2
        elif nh==64:
            assert boxsize==16
        elif nh==128:
            assert boxsize==8
        else:
            print(f'boxsize {boxsize} not supported yet')

        l = models.utils.LinearWN
        c = models.utils.ConvTranspose2dWN
        v = models.utils.Conv2dWN
        a = nn.LeakyReLU
        s = nn.Sequential

        self.encmod = s(v(16, 16, 1, 1, 0), a(0.2, inplace=True))
        inch = 16 + 16 if enable_id_encoder else 16
        if self.viewcond:
            self.viewmod = s(l( 3,    16), a(0.2, inplace=True),
                             l(16, 8*4*4), a(0.2, inplace=True))
            inch += 8

        if imsize == 1024:
            size = [inch, 256, 128, 128, 64, 64, 32, 16, self.boxsize * self.outch]
            scale_factor = 4
        elif imsize == 512:
            size = [inch, 256, 128, 128, 64, 64, 32, self.boxsize * self.outch]
            scale_factor = 2
        else:
            print(f'Unsupported image size: {size}')
            quit()
        self.nlayers = len(size)-1

        h = 8
        for i in range(self.nlayers):
            #t = [c(size[i], size[i+1], 4, 2, 1)]
            t = [c(size[i], size[i+1], 4, 2, 1)]
            h *= 2

            if i < self.nlayers-1:
                t.append(a(0.2, inplace=True))
            self.add_module(f't{i}', s(*t))

        if self.texwarp:
            raise NotImplementedError('TexWarp is not implemented yet')
            self.warpmod = s(v(inch, 256, 1, 1, 0), a(0.2, inplace=True),
                             c( 256, 256, 4, 2, 1), a(0.2, inplace=True),
                             c( 256, 128, 4, 2, 1), a(0.2, inplace=True),
                             c( 128, 128, 4, 2, 1), a(0.2, inplace=True),
                             c( 128,  64, 4, 2, 1), a(0.2, inplace=True),
                             c(  64,  64, 4, 2, 1), a(0.2, inplace=True),
                             c(  64,   2, 4, 2, 1),
                             nn.Upsample(scale_factor=scale_factor, mode='bilinear'))

        if self.viewcond:
            models.utils.initseq(self.viewmod)
        models.utils.initseq(self.encmod)
        for i in range(self.nlayers):
            models.utils.initseq(self._modules[f't{i}'])
        if self.texwarp:
            models.utils.initseq(self.warpmod)

        # self.bias = nn.Parameter(torch.zeros(self.boxsize * self.outch, imsize, imsize))
        # self.bias.data.zero_()

        xgrid, ygrid = np.meshgrid(np.linspace(-1.0, 1.0, imsize),
                                   np.linspace(-1.0, 1.0, imsize))
        grid = np.concatenate((xgrid[None, :, :],
                               ygrid[None, :, :]),
                              axis=0)[None, ...].astype(np.float32)
        self.register_buffer("warpbias", torch.from_numpy(grid), persistent=False)


    def forward(self, ex_enc, id_enc, id_gainbias, view, use_warp = True, iternum = -1):

        z = self.encmod(ex_enc).view(-1, 16, 4, 4)
        x = torch.cat([z, id_enc], dim=1) if id_enc is not None else z

        if self.viewcond:
            v = self.viewmod(view).view(-1, 8, 4, 4)
            x = torch.cat([v, x], dim=1)
        x_orig = x

        ###############################################################################################################################
        scale = 1 / sqrt(2)
        for i in range(self.nlayers):
            xx = self._modules[f't{i}'](x)

            if id_gainbias is not None:
                n = id_gainbias[i].shape[1] // 2
                if n == xx.shape[1]:
                    x = (xx * (id_gainbias[i][:,:n,...]*0.01 + 1.0) + id_gainbias[i][:,n:,...]) * scale
                elif n*2 == xx.shape[1]:
                    x = (xx + id_gainbias[i]) * scale
                else:
                    x = xx #note: last layer (1024x1024) ignores the pass through since slab is larger than 3 channels
            else:
                x = xx

        ###############################################################################################################################

        if self.texwarp and use_warp:
            w = self.warpmod(x_orig)
            w = w / self.imsize + self.warpbias
            x = torch.nn.functional.grid_sample(x, w.permute(0, 2, 3, 1))
        else:
            w = None
        # tex = x + self.bias[None, :, :, :]
        tex = x

        x0 = tex
        x = tex
        h = int(np.sqrt(self.nboxes))
        w = int(h)
        x = x.view(x.size(0), self.boxsize, self.outch, h, self.boxsize, w, self.boxsize)
        x = x.permute(0, 3, 5, 2, 1, 4, 6)

        x = x.reshape(x.size(0), self.nboxes, self.outch, self.boxsize, self.boxsize, self.boxsize)

        return x


###############################################################################
#add per-pixel gains as well as bias
#add geometry and motion as output, remove warp option, remove viewcond option
#positivity for alpha
class DecoderGeoSlab2(nn.Module):
    def __init__(self, uv, tri, uvtri, nvtx, motion_size, geo_size, imsize, nboxes, boxsize, enable_id_encoder=True, cfg=None):
        super(DecoderGeoSlab2, self).__init__()


        assert(motion_size < imsize)
        assert(geo_size < imsize)

        self.cfg = cfg
        self.motion_size = motion_size
        self.geo_size = geo_size
        self.imsize = imsize
        self.nboxes = nboxes
        self.boxsize = boxsize


        nh = int(np.sqrt(self.nboxes))
        assert nh*nh==self.nboxes
        if nh==512:
            assert boxsize==2
        elif nh==64:
            assert boxsize==16
        elif nh==128:
            assert boxsize==8
        else:
            print(f'boxsize {boxsize} not supported yet')

        l = models.utils.LinearWN
        c = models.utils.ConvTranspose2dWN
        v = models.utils.Conv2dWN
        a = nn.LeakyReLU
        s = nn.Sequential

        #reduce noise effect of latent expression code
        self.encmod = s(v(16, 16, 1, 1, 0), a(0.2, inplace=True))
        models.utils.initseq(self.encmod)

        inch = 16 + 16 if enable_id_encoder else 16 #first is for expression, second for identity

        if imsize == 1024:
            size = [inch, 256, 128, 128, 64, 64, 32, 16, self.boxsize]
            scale_factor = 4
        elif imsize == 512:
            size = [inch, 256, 128, 128, 64, 64, 32, self.boxsize]
            scale_factor = 2
        else:
            print(f'Unsupported image size: {size}')
            quit()
        self.nlayers = len(size)-1

        #build deconv arch with early exists for geometry and motion
        h = 8
        for i in range(self.nlayers):
            # t = [c(size[i], size[i+1], h, h, 4, 2, 1)]
            t = [c(size[i], size[i+1], 4, 2, 1)]
            if i < self.nlayers-1:
                t.append(a(0.2, inplace=True))
            self.add_module(f't{i}', s(*t))
            models.utils.initseq(self._modules[f't{i}'])

            if h == motion_size:
                self.motion = s(v(size[i+1], 64, 1, 1, 0), a(0.2, inplace=True), v(64, 9, 1, 1, 0))
                models.utils.initseq(self.motion)

            if h == geo_size:
                self.geo = s(v(size[i+1], 64, 1, 1, 0), a(0.2, inplace=True), v(64, 3, 1, 1, 0))
                models.utils.initseq(self.geo)

            h *= 2


        # self.bias = nn.Parameter(torch.zeros(self.boxsize, imsize, imsize))
        # self.bias.data.zero_()

        xgrid, ygrid = np.meshgrid(np.linspace(-1.0, 1.0, imsize),
                                   np.linspace(-1.0, 1.0, imsize))
        grid = np.concatenate((xgrid[None, :, :],
                               ygrid[None, :, :]),
                              axis=0)[None, ...].astype(np.float32)
        self.register_buffer("warpbias", torch.from_numpy(grid), persistent=False)



        #create cropping coordinates for geometry points
        vlists = [list() for _ in range(nvtx)]

        try:
            for fi in range(tri.shape[0]):
                for fv in range(3):
                    vlists[tri[fi,fv]].append(uvtri[fi,fv])
        except IndexError:
            print(f"{fi=}")
            print(f"{fv=}")
            print(f"{tri[fi,fv]=}")
            print(f"{uvtri[fi,fv]=}")
            raise
        nMaxUVsPerVertex = np.max([len(v) for v in vlists])
        print('Max UVs per vertex: {}'.format(nMaxUVsPerVertex))
        nMaxUVsPerVertex = 1#2
        uvspervert = np.zeros((nvtx, nMaxUVsPerVertex), dtype=np.int32)
        wuvspervert = np.zeros((nvtx, nMaxUVsPerVertex), dtype=np.float32)
        uvmask = np.ones((nvtx,), dtype=np.float32)
        for tvi in range(len(vlists)):
            if not (len(vlists[tvi])):
                uvmask[tvi] = 0
                continue
            for vsi in range(nMaxUVsPerVertex):
                if vsi<len(vlists[tvi]):
                    uvspervert[tvi,vsi] = vlists[tvi][vsi]
                    wuvspervert[tvi,vsi] = 1.0/nMaxUVsPerVertex
                elif len(vlists[tvi]):
                    uvspervert[tvi,vsi] = vlists[tvi][0]
                    wuvspervert[tvi,vsi] = 1.0/nMaxUVsPerVertex
        self.register_buffer("t_nl_uvspervert",
                             torch.from_numpy(uvspervert).long().to("cuda"))
        self.register_buffer("t_nl_wuvspervert",
                             torch.from_numpy(wuvspervert).to("cuda"))
        t_nl_geom_vert_uvs = torch.from_numpy(uv).to("cuda")[self.t_nl_uvspervert,:]
        coords = t_nl_geom_vert_uvs.view(1, -1, nMaxUVsPerVertex, 2) * 2 - 1.0
        self.register_buffer("coords", coords)


    def forward(self, ex_enc, id_enc, id_gainbias, iternum = -1):

        z = self.encmod(ex_enc).view(-1, 16, 4, 4)
        x = torch.cat([z, id_enc], dim=1) if id_enc is not None else z

        scale = 1 / sqrt(2)
        for i in range(self.nlayers):
            xx = self._modules[f't{i}'](x)

            if id_gainbias is not None:
                n = id_gainbias[i].shape[1] // 2
                if n == xx.shape[1]:
                    x = (xx * (id_gainbias[i][:,:n,...]*0.1 + 1.0) + id_gainbias[i][:,n:,...]) * scale
                elif n*2 == xx.shape[1]:
                    x = (xx + id_gainbias[i]) * scale
                else:
                    x = xx #note: last layer (1024x1024) ignores the pass through since slab is larger than 3 channels
            else:
                x = xx

            if x.shape[-1] == self.motion_size:
                mot = self.motion(x)
            if x.shape[-1] == self.geo_size:
                geo = self.geo(x)

        # TODO: add trunc_exp for stability
        tex = torch.exp((x) * 0.1)

        #get motion
        mot = mot.view(mot.size(0), 9, -1).permute(0, 2, 1).contiguous()
        primposresid = mot[:, :, 0:3] * 0.01
        primrvecresid = mot[:, :, 3:6] * 0.01
        primscaleresid = torch.exp(0.01 * mot[:, :, 6:9])

        #get geometry
        coords = self.coords.expand((geo.size(0), -1, -1, -1))
        geo = F.grid_sample(geo, coords).mean(dim=3).permute(0, 2, 1)


        x0 = tex
        x = tex
        h = int(np.sqrt(self.nboxes))
        w = int(h)
        x = x.view(x.size(0), self.boxsize, 1, h, self.boxsize, w, self.boxsize)
        x = x.permute(0, 3, 5, 2, 1, 4, 6)

        x = x.reshape(x.size(0), self.nboxes, 1, self.boxsize, self.boxsize, self.boxsize)

        return x, geo, primposresid, primrvecresid, primscaleresid
