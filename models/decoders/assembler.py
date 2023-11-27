"""
"""
import os
import time
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.utils
from models.decoders.geometry import GeometryDecoder
from models.decoders.rgb import RGBDecoder


class Assembler(nn.Module):
    def __init__(
        self,
        vt,
        vi,
        vti,
        vertmean,
        vertstd,
        volradius,  # dectype="deconv",
        nprims=128 * 128,
        primsize=(8, 8, 8),
        alphafade=False,
        postrainstart=0,
    ):
        super(Assembler, self).__init__()

        self.volradius = volradius
        self.alphafade = alphafade
        self.postrainstart = postrainstart

        self.nprims = nprims
        self.primsize = primsize

        self.rodrig = models.utils.Rodrigues()

        # payload decoder
        imsize = int(sqrt(nprims)) * primsize[1]
        self.rgbdec = RGBDecoder(imsize=imsize, nboxes=nprims, boxsize=primsize[0], outch=3, viewcond=True)
        self.geodec = GeometryDecoder(
            vt,
            vi,
            vti,
            nvtx=vertmean.shape[-2],
            motion_size={256: 16, 16384: 128}[nprims],
            geo_size=256,
            imsize=imsize,
            nboxes=nprims,
            boxsize=primsize[0],
        )

        self.register_buffer("vt", torch.tensor(vt), persistent=False)
        self.register_buffer("vertmean", vertmean, persistent=False)
        self.vertstd = vertstd

        # apath=os.getenv('RSC_AVATAR_RSCASSET_PATH')
        apath = "/checkpoint/avatar/jinkyuk/rsc-assets"
        idximpath = f"{apath}/idxmap"
        # idximpath = "/mnt/captures/stephenlombardi/idxmap"
        self.register_buffer(
            "idxim", torch.tensor(np.load("{}/retop_idxim_1024.npy".format(idximpath))).long(), persistent=False
        )
        self.register_buffer(
            "tidxim", torch.tensor(np.load("{}/retop_tidxim_1024.npy".format(idximpath))).long(), persistent=False
        )
        self.register_buffer(
            "barim", torch.tensor(np.load("{}/retop_barim_1024.npy".format(idximpath))), persistent=False
        )

        self.register_buffer("adaptwarps", 0 * torch.ones(self.nprims))

    def forward(
        self,
        gt_geo,  #########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        id_cond,
        encoding,
        viewpos,
        condinput=None,
        renderoptions={},
        trainiter=-1,
        losslist=[],
    ):
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

        # TODO:
        if condinput is not None:
            encoding = torch.cat([encoding, condinput], dim=1)

        z_geo, b_geo = (id_cond["z_geo"], id_cond["b_geo"])
        primalpha, geo, primposresid, primrvecresid, primscaleresid = self.geodec(
            encoding, z_geo, b_geo, iternum=trainiter
        )

        geo = geo * self.vertstd + self.vertmean

        ##########################################################################################################
        geo_orig = geo
        if trainiter <= self.postrainstart:
            geo = gt_geo * self.vertstd + self.vertmean
        ##########################################################################################################

        # NOTE(julieta) do not do this, this is very slow :(
        # postex = torch.stack([
        #     self.barim[:, :, 0, None] * geo[i, self.idxim[:, :, 0], :] +
        #     self.barim[:, :, 1, None] * geo[i, self.idxim[:, :, 1], :] +
        #     self.barim[:, :, 2, None] * geo[i, self.idxim[:, :, 2], :]
        #     for i in range(geo.size(0))], dim=0).permute(0, 3, 1, 2) / self.volradius
        postex = (
            self.barim[:, :, 0, None] * geo.index_select(1, self.idxim[:, :, 0].reshape(-1)).reshape(-1, 1024, 1024, 3)
            + self.barim[:, :, 1, None]
            * geo.index_select(1, self.idxim[:, :, 1].reshape(-1)).reshape(-1, 1024, 1024, 3)
            + self.barim[:, :, 2, None]
            * geo.index_select(1, self.idxim[:, :, 2].reshape(-1)).reshape(-1, 1024, 1024, 3)
        ).permute(0, 3, 1, 2) / self.volradius

        # placement of primitives on mesh
        if self.nprims == 1:
            primpos = torch.zeros((postex.size(0), 1, 3), device=postex.device)
            primscale = 2.0

            v0 = torch.tensor([0.0, 0.0, 0.0], device=postex.device)[None, None, None, :].repeat(
                postex.size(0), 1, 1, 1
            )
            v1 = torch.tensor([1.0, 0.0, 0.0], device=postex.device)[None, None, None, :].repeat(
                postex.size(0), 1, 1, 1
            )
            v2 = torch.tensor([0.0, 1.0, 0.0], device=postex.device)[None, None, None, :].repeat(
                postex.size(0), 1, 1, 1
            )
        elif self.nprims == 8:
            primpos = postex[:, :, 256::512, 128::256].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 4.0

            v0 = geo[:, self.idxim[256::512, 128::256, 0], :]  # vert 0?
            v1 = geo[:, self.idxim[256::512, 128::256, 1], :]  # vert 1?
            v2 = geo[:, self.idxim[256::512, 128::256, 2], :]  # vert 2?
            vt0 = self.vt[self.tidxim[256::512, 128::256, 0], :]
            vt1 = self.vt[self.tidxim[256::512, 128::256, 1], :]
            vt2 = self.vt[self.tidxim[256::512, 128::256, 2], :]
        elif self.nprims == 64:
            primpos = postex[:, :, 64::128, 64::128].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 8.0

            v0 = geo[:, self.idxim[64::128, 64::128, 0], :]  # vert 0?
            v1 = geo[:, self.idxim[64::128, 64::128, 1], :]  # vert 1?
            v2 = geo[:, self.idxim[64::128, 64::128, 2], :]  # vert 2?
            vt0 = self.vt[self.tidxim[64::128, 64::128, 0], :]
            vt1 = self.vt[self.tidxim[64::128, 64::128, 1], :]
            vt2 = self.vt[self.tidxim[64::128, 64::128, 2], :]
        elif self.nprims == 256:
            primpos = postex[:, :, 32::64, 32::64].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 12.0

            ###################################################
            if trainiter < 100:
                with torch.no_grad():
                    centdiffx = postex[:, :, 32::64, (32 + 64) :: 64] - postex[:, :, 32::64, 32:-64:64]
                    centdiffx = torch.cat((centdiffx, centdiffx[:, :, :, -1:]), dim=3)
                    centdiffy = postex[:, :, (32 + 64) :: 64, 32::64] - postex[:, :, 32:-64:64, 32::64]
                    centdiffy = torch.cat((centdiffy, centdiffy[:, :, -1:, :]), dim=2)
                    centdiffx = centdiffx.norm(dim=1)
                    centdiffy = centdiffy.norm(dim=1)
                    centsize = torch.max(centdiffx, centdiffy)
                    centsize = torch.max(centsize, dim=0)[0].view(self.nprims)
                    warps_vec = 2 / centsize
                    if self.adaptwarps.max().item() == 0:
                        self.adaptwarps.data[:] = warps_vec
                    else:
                        self.adaptwarps.data[:] = self.adaptwarps * 0.9 + 0.1 * warps_vec
            primscale = self.adaptwarps[None, :, None] * 0.8
            ###################################################

            v0 = geo[:, self.idxim[32::64, 32::64, 0], :]  # vert 0?
            v1 = geo[:, self.idxim[32::64, 32::64, 1], :]  # vert 1?
            v2 = geo[:, self.idxim[32::64, 32::64, 2], :]  # vert 2?
            vt0 = self.vt[self.tidxim[32::64, 32::64, 0], :]
            vt1 = self.vt[self.tidxim[32::64, 32::64, 1], :]
            vt2 = self.vt[self.tidxim[32::64, 32::64, 2], :]

            geodu = postex[:, :, :, 1:] - postex[:, :, :, :-1]
            geodv = postex[:, :, 1:, :] - postex[:, :, :-1, :]
            vcenterdu = geodu[:, :, 32::64, 32::64].permute(0, 2, 3, 1)
            vcenterdv = geodv[:, :, 32::64, 32::64].permute(0, 2, 3, 1)  # match v0 (channels last)

        elif self.nprims == 512:
            primpos = postex[:, :, 32::64, 16::32].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 16.0

            v0 = geo[:, self.idxim[32::64, 16::32, 0], :]  # vert 0?
            v1 = geo[:, self.idxim[32::64, 16::32, 1], :]  # vert 1?
            v2 = geo[:, self.idxim[32::64, 16::32, 2], :]  # vert 2?
            vt0 = self.vt[self.tidxim[32::64, 16::32, 0], :]
            vt1 = self.vt[self.tidxim[32::64, 16::32, 1], :]
            vt2 = self.vt[self.tidxim[32::64, 16::32, 2], :]
        elif self.nprims == 4096:
            primpos = postex[:, :, 8::16, 8::16].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 32.0

            v0 = geo[:, self.idxim[8::16, 8::16, 0], :]  # vert 0?
            v1 = geo[:, self.idxim[8::16, 8::16, 1], :]  # vert 1?
            v2 = geo[:, self.idxim[8::16, 8::16, 2], :]  # vert 2?
            vt0 = self.vt[self.tidxim[8::16, 8::16, 0], :]
            vt1 = self.vt[self.tidxim[8::16, 8::16, 1], :]
            vt2 = self.vt[self.tidxim[8::16, 8::16, 2], :]
        elif self.nprims == 16384:
            # primpos = postex[:, :, 4::8, 4::8].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            # primscale = 48.

            # v0 = geo[:, self.idxim[4::8, 4::8, 0], :] # vert 0?
            # v1 = geo[:, self.idxim[4::8, 4::8, 1], :] # vert 1?
            # v2 = geo[:, self.idxim[4::8, 4::8, 2], :] # vert 2?
            # vt0 = self.vt[self.tidxim[4::8, 4::8, 0], :]
            # vt1 = self.vt[self.tidxim[4::8, 4::8, 1], :]
            # vt2 = self.vt[self.tidxim[4::8, 4::8, 2], :]

            primpos = postex[:, :, 4::8, 4::8].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 48.0

            ###################################################
            if trainiter < 100:
                with torch.no_grad():
                    centdiffx = postex[:, :, 4::8, (4 + 8) :: 8] - postex[:, :, 4::8, 4:-8:8]
                    centdiffx = torch.cat((centdiffx, centdiffx[:, :, :, -1:]), dim=3)
                    centdiffy = postex[:, :, (4 + 8) :: 8, 4::8] - postex[:, :, 4:-8:8, 4::8]
                    centdiffy = torch.cat((centdiffy, centdiffy[:, :, -1:, :]), dim=2)
                    centdiffx = centdiffx.norm(dim=1)
                    centdiffy = centdiffy.norm(dim=1)
                    centsize = torch.max(centdiffx, centdiffy)
                    centsize = torch.max(centsize, dim=0)[0].view(self.nprims)
                    warps_vec = 2 / centsize
                    if self.adaptwarps.max().item() == 0:
                        self.adaptwarps.data[:] = warps_vec
                    else:
                        self.adaptwarps.data[:] = self.adaptwarps * 0.9 + 0.1 * warps_vec
            primscale = self.adaptwarps[None, :, None] * 0.8
            ###################################################

            v0 = geo[:, self.idxim[4::8, 4::8, 0], :]  # vert 0?
            v1 = geo[:, self.idxim[4::8, 4::8, 1], :]  # vert 1?
            v2 = geo[:, self.idxim[4::8, 4::8, 2], :]  # vert 2?
            vt0 = self.vt[self.tidxim[4::8, 4::8, 0], :]
            vt1 = self.vt[self.tidxim[4::8, 4::8, 1], :]
            vt2 = self.vt[self.tidxim[4::8, 4::8, 2], :]

            geodu = postex[:, :, :, 1:] - postex[:, :, :, :-1]
            geodv = postex[:, :, 1:, :] - postex[:, :, :-1, :]
            vcenterdu = geodu[:, :, 4::8, 4::8].permute(0, 2, 3, 1)
            vcenterdv = geodv[:, :, 4::8, 4::8].permute(0, 2, 3, 1)  # match v0 (channels last)

        elif self.nprims == 32768:
            primpos = postex[:, :, 4::8, 2::4].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 64.0

            v0 = geo[:, self.idxim[4::8, 2::4, 0], :]  # vert 0?
            v1 = geo[:, self.idxim[4::8, 2::4, 1], :]  # vert 1?
            v2 = geo[:, self.idxim[4::8, 2::4, 2], :]  # vert 2?
            vt0 = self.vt[self.tidxim[4::8, 2::4, 0], :]
            vt1 = self.vt[self.tidxim[4::8, 2::4, 1], :]
            vt2 = self.vt[self.tidxim[4::8, 2::4, 2], :]
        elif self.nprims == 262144:
            primpos = postex[:, :, 1::2, 1::2].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 128.0

            v0 = geo[:, self.idxim[1::2, 1::2, 0], :]  # vert 0?
            v1 = geo[:, self.idxim[1::2, 1::2, 1], :]  # vert 1?
            v2 = geo[:, self.idxim[1::2, 1::2, 2], :]  # vert 2?
            vt0 = self.vt[self.tidxim[1::2, 1::2, 0], :]
            vt1 = self.vt[self.tidxim[1::2, 1::2, 1], :]
            vt2 = self.vt[self.tidxim[1::2, 1::2, 2], :]
        else:
            raise

        # # compute TBN matrix
        # v01 = v1 - v0
        # v02 = v2 - v0
        # vt01 = vt1 - vt0
        # vt02 = vt2 - vt0
        # f = 1. / (vt01[None, :, :, 0] * vt02[None, :, :, 1] - vt01[None, :, :, 1] * vt02[None, :, :, 0])
        # tangent = f[:, :, :, None] * torch.stack([
        #     v01[:, :, :, 0] * vt02[None, :, :, 1] - v02[:, :, :, 0] * vt01[None, :, :, 1],
        #     v01[:, :, :, 1] * vt02[None, :, :, 1] - v02[:, :, :, 1] * vt01[None, :, :, 1],
        #     v01[:, :, :, 2] * vt02[None, :, :, 1] - v02[:, :, :, 2] * vt01[None, :, :, 1]], dim=-1)
        # tangent = tangent / torch.sqrt(torch.sum(tangent ** 2, dim=-1, keepdim=True)).clamp(min=1e-5)
        # normal = torch.cross(v01, v02, dim=3)
        # normal = normal / torch.sqrt(torch.sum(normal ** 2, dim=-1, keepdim=True)).clamp(min=1e-5)
        # bitangent = torch.cross(normal, tangent, dim=3)
        # bitangent = bitangent / torch.sqrt(torch.sum(bitangent ** 2, dim=-1, keepdim=True)).clamp(min=1e-5)

        # # set orientation
        # if True:#self.flipy:
        #     primrot = torch.stack((tangent, -bitangent, normal), dim=-2)
        # else:
        #     primrot = torch.stack((tangent, bitangent, normal), dim=-2)
        # primrot = primrot.view(encoding.size(0), -1, 3, 3).contiguous().permute(0, 1, 3, 2).contiguous()

        tangent = vcenterdu
        tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-8)
        normal = torch.cross(tangent, vcenterdv, dim=3)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True).clamp(min=1e-8)
        bitangent = torch.cross(normal, tangent, dim=3)
        bitangent = bitangent / torch.norm(bitangent, dim=-1, keepdim=True).clamp(min=1e-8)
        primrot = (
            torch.stack((tangent, bitangent, normal), dim=-2)
            .view(encoding.size(0), -1, 3, 3)
            .contiguous()
            .permute(0, 1, 3, 2)
            .contiguous()
        )

        # primposresid, primrvecresid, primscaleresid = self.motiondec(encoding)
        if trainiter <= self.postrainstart:
            # if True: #########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            primposresid = primposresid * 0.0
            primrvecresid = primrvecresid * 0.0
            primscaleresid = primscaleresid * 0.0 + 1.0
        elif trainiter <= 2 * self.postrainstart:
            weight = (2 * self.postrainstart - trainiter) / (self.postrainstart + 1)
            primposresid = primposresid * (1 - weight) + weight * 0.0
            primrvecresid = primrvecresid * (1 - weight) + weight * 0.0
            primscaleresid = primscaleresid * (1 - weight) + weight * 1.0
            # print(weight)
        # else:
        #     weight = 1 / (1 + trainiter - self.postrainstart)**0.25
        #     primposresid = primposresid * (1-weight) + weight * 0.
        #     primrvecresid = primrvecresid * (1-weight) + weight * 0.
        #     primscaleresid = primscaleresid * (1-weight) + weight * 1.
        #     print(weight)
        primrotresid = self.rodrig(primrvecresid.view(-1, 3)).view(encoding.size(0), nprims, 3, 3)

        primpos = primpos + primposresid
        primrot = torch.bmm(primrot.view(-1, 3, 3), primrotresid.view(-1, 3, 3)).view(encoding.size(0), nprims, 3, 3)
        primscale = primscale * primscaleresid

        viewdirs = viewpos / torch.sqrt(torch.sum(viewpos**2, dim=1, keepdim=True))
        # if self.rgbadec is not None:
        #     # TODO: let scale, bias be an input argument?
        #     scale = torch.tensor([25., 25., 25., 1.], device=encoding.device)
        #     bias = torch.tensor([100., 100., 100., 0.], device=encoding.device)
        #     primrgba = F.relu(bias[None, None, :, None, None, None] + scale[None, None, :, None, None, None] *
        #             self.rgbadec(torch.cat([encoding, viewdirs], dim=1), trainiter=trainiter))
        # else:
        #     primrgb = F.relu(100. + 25. * self.rgbdec(torch.cat([encoding, viewdirs], dim=1), trainiter=trainiter))
        #     primalpha = F.relu(self.alphadec(encoding, trainiter=trainiter))
        # template = torch.cat([primrgb, primalpha], dim=2)

        z_tex, b_tex = (id_cond["z_tex"], id_cond["b_tex"]) if not self.disable_id_encoder else (None, None)
        primrgb = self.rgbdec(encoding, z_tex, b_tex, view=viewdirs, use_warp=False, iternum=trainiter)

        # primalpha = self.alphadec(z_ex, id_cond['z_geo'], id_cond['b_geo'], view=None, use_warp=False, iternum=trainiter)
        template = torch.cat([F.relu(primrgb * 25.0 + 100.0), F.relu(primalpha)], dim=2)

        # primrgb = F.relu(100. + 25. * self.rgbdec(torch.cat([encoding, viewdirs], dim=1), trainiter=trainiter)) ######################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # primalpha = F.relu(self.alphadec(encoding, trainiter=trainiter))
        # template = torch.cat([primrgb, primalpha], dim=2)

        if self.alphafade:
            gridz, gridy, gridx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, self.primsize, device=encoding.device),
                torch.linspace(-1.0, 1.0, self.primsize, device=encoding.device),
                torch.linspace(-1.0, 1.0, self.primsize, device=encoding.device),
            )
            grid = torch.stack([gridx, gridy, gridz], dim=-1)
            template = (
                template
                * torch.stack(
                    [
                        gridx * 0.0 + 1.0,
                        gridy * 0.0 + 1.0,
                        gridz * 0.0 + 1.0,
                        torch.exp(-8.0 * torch.sum(grid**8, dim=-1)),
                    ],
                    dim=0,
                )[None, None, :, :, :, :]
            )

        if "viewaxes" in renderoptions and renderoptions["viewaxes"]:
            template[
                :,
                :,
                3,
                template.size(3) // 2 : template.size(3) // 2 + 1,
                template.size(4) // 2 : template.size(4) // 2 + 1,
                :,
            ] = 2550.0
            template[
                :,
                :,
                0,
                template.size(3) // 2 : template.size(3) // 2 + 1,
                template.size(4) // 2 : template.size(4) // 2 + 1,
                :,
            ] = 2550.0
            template[
                :,
                :,
                3,
                template.size(3) // 2 : template.size(3) // 2 + 1,
                :,
                template.size(5) // 2 : template.size(5) // 2 + 1,
            ] = 2550.0
            template[
                :,
                :,
                1,
                template.size(3) // 2 : template.size(3) // 2 + 1,
                :,
                template.size(5) // 2 : template.size(5) // 2 + 1,
            ] = 2550.0
            template[
                :,
                :,
                3,
                :,
                template.size(4) // 2 : template.size(4) // 2 + 1,
                template.size(5) // 2 : template.size(5) // 2 + 1,
            ] = 2550.0
            template[
                :,
                :,
                2,
                :,
                template.size(4) // 2 : template.size(4) // 2 + 1,
                template.size(5) // 2 : template.size(5) // 2 + 1,
            ] = 2550.0

        if "colorprims" in renderoptions and renderoptions["colorprims"]:
            lightdir = torch.tensor([1.0, 1.0, 1.0], device=template.device)
            lightdir = lightdir / torch.sqrt(torch.sum(lightdir**2))
            zz, yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, template.size(-1), device=template.device),
                torch.linspace(-1.0, 1.0, template.size(-1), device=template.device),
                torch.linspace(-1.0, 1.0, template.size(-1), device=template.device),
            )
            primnormalx = torch.where(
                (torch.abs(xx) >= torch.abs(yy)) & (torch.abs(xx) >= torch.abs(zz)),
                torch.sign(xx) * torch.ones_like(xx),
                torch.zeros_like(xx),
            )
            primnormaly = torch.where(
                (torch.abs(yy) >= torch.abs(xx)) & (torch.abs(yy) >= torch.abs(zz)),
                torch.sign(yy) * torch.ones_like(xx),
                torch.zeros_like(xx),
            )
            primnormalz = torch.where(
                (torch.abs(zz) >= torch.abs(xx)) & (torch.abs(zz) >= torch.abs(yy)),
                torch.sign(zz) * torch.ones_like(xx),
                torch.zeros_like(xx),
            )
            primnormal = torch.stack([primnormalx, primnormaly, primnormalz], dim=-1)
            primnormal = primnormal / torch.sqrt(torch.sum(primnormal**2, dim=-1, keepdim=True))
            template[:, :, 3, :, :, :] = 1000.0
            np.random.seed(123456)
            for i in range(template.size(1)):
                template[:, i, 0, :, :, :] = np.random.rand() * 255.0
                template[:, i, 1, :, :, :] = np.random.rand() * 255.0
                template[:, i, 2, :, :, :] = np.random.rand() * 255.0

                lightdir0 = torch.sum(primrot[:, i, :, :] * lightdir[None, :, None], dim=-2)
                template[:, i, :3, :, :, :] *= 1.2 * torch.sum(lightdir0[:, None, None, None, :] * primnormal, dim=-1)[
                    :, None, :, :, :
                ].clamp(min=0.05)

        # visualize slab
        if "viewslab" in renderoptions and renderoptions["viewslab"]:
            yy, xx = torch.meshgrid(
                torch.linspace(0.9, -0.9, 16, device=template.device),
                torch.linspace(-0.9, 0.9, 16, device=template.device),
            )
            primpos = (
                torch.stack([xx, yy, xx * 0.0], dim=-1)[None, :, :, :]
                .repeat(template.size(0), 1, 1, 1)
                .view(-1, 256, 3)
            )
            primrot = torch.eye(3, device=template.device)[None, None, :, :].repeat(template.size(0), 256, 1, 1)
            primscale = torch.ones((template.size(0), 256, 3), device=template.device) * 16.0

        losses = {}

        # if "primvol" in losslist:
        #    losses["primvol"] = torch.mean(torch.prod(1. / primscale, dim=-1), dim=-1)
        if "primvolsum" in losslist:
            losses["primvolsum"] = torch.sum(torch.prod(1.0 / primscale, dim=-1), dim=-1)
        # if "primtoffset" in losslist:
        #    losses["primtoffset"] = torch.mean(torch.sum(primposresid.view(primposresid.size(0), -1) ** 2, dim=-1)) * \
        #            1000. / (1000. + 0.5 * trainiter)
        # if "primroffset" in losslist:
        #    losses["primroffset"] = torch.mean(torch.sum(primrvecresid.view(primrvecresid.size(0), -1) ** 2, dim=-1)) * \
        #            1000. / (1000. + 0.5 * trainiter)

        #######################################################################################################
        if trainiter <= self.postrainstart:
            geo = geo_orig
        #######################################################################################################

        return {
            "verts": geo,
            "template": template,
            "primpos": primpos,
            "primrot": primrot,
            "primscale": primscale,
        }, losses
