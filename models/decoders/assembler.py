import time
from math import sqrt
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.utils
from models.decoders.geometry import GeometryDecoder
from models.decoders.rgb import RGBDecoder


class DecoderAssembler(nn.Module):
    def __init__(
        self,
        vt: np.ndarray,
        vi: np.ndarray,
        vti: np.ndarray,
        vertmean: torch.Tensor,
        vertstd: torch.Tensor,
        volradius: float,
        nprims: int = 128 * 128,
        primsize: Tuple[int, int, int] = (8, 8, 8),
        postrainstart: int = 100,
    ):
        super(DecoderAssembler, self).__init__()

        self.volradius = volradius

        self.nprims = nprims
        self.primsize = primsize
        self.postrainstart = postrainstart
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

        # TODO(julieta) compute these on the fly, or commit them to the repo -- 30MB tho!
        idximpath = f"assets/rsc-assets/idxmap"
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
        id_cond: Dict[str, torch.Tensor],
        encoding: torch.Tensor,
        viewpos: torch.Tensor,
        condinput=None,
        trainiter=-1,  # TODO(julieta) do not change the forward pass based on the training iteration
        loss_set=set(),
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Parameters
        ----------
        encoding : [B, 256] Expression encoding of current frame
        id_cond: Dictionary with geometry and texture identity biases and code
        viewpos: Viewing position of target camera view
        condinput: Additional conditioning input (e.g., headpose) # TODO(julieta maybe remove?)
        trainiter: Current training iteration
        loss_set: Set of losses to compute and return
        Returns
        -------
        result : Tuple of two dictionaries.
            The first dictionary contains predicted geometry (vertex positions), primitive contents (rgb and alpha) as
            well as locations, scaling, and rotations.
            The second dictionary contains any requested losses.
        """
        nprims = self.nprims

        # TODO:
        if condinput is not None:
            encoding = torch.cat([encoding, condinput], dim=1)

        z_geo, b_geo = id_cond["z_geo"], id_cond["b_geo"]
        primalpha, geo, primposresid, primrvecresid, primscaleresid = self.geodec(encoding, z_geo, b_geo)
        geo = geo * self.vertstd + self.vertmean

        # NOTE(julieta) do not do this, this is very slow :(
        # postex = torch.stack([
        #     self.barim[:, :, 0, None] * geo[i, self.idxim[:, :, 0], :] +
        #     self.barim[:, :, 1, None] * geo[i, self.idxim[:, :, 1], :] +
        #     self.barim[:, :, 2, None] * geo[i, self.idxim[:, :, 2], :]
        #     for i in range(geo.size(0))], dim=0).permute(0, 3, 1, 2) / self.volradius
        # fmt: off
        postex = (
            self.barim[:, :, 0, None] * geo.index_select(1, self.idxim[:, :, 0].reshape(-1)).reshape(-1, 1024, 1024, 3) +
            self.barim[:, :, 1, None] * geo.index_select(1, self.idxim[:, :, 1].reshape(-1)).reshape(-1, 1024, 1024, 3) + 
            self.barim[:, :, 2, None] * geo.index_select(1, self.idxim[:, :, 2].reshape(-1)).reshape(-1, 1024, 1024, 3)
        ).permute(0, 3, 1, 2) / self.volradius
        # fmt: on

        vcenterdu, vcenterdv = None, None

        # TODO(julieta) This logic effectively only works for 256 and 16384 primitives. We should generalize, extend and
        # test it so it works on the passed number of primitives

        # Placement of primitives on mesh
        if self.nprims == 1:
            primpos = torch.zeros((postex.size(0), 1, 3), device=postex.device)
            primscale = 2.0

        elif self.nprims == 8:
            primpos = postex[:, :, 256::512, 128::256].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 4.0

        elif self.nprims == 64:
            primpos = postex[:, :, 64::128, 64::128].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 8.0

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

            geodu = postex[:, :, :, 1:] - postex[:, :, :, :-1]
            geodv = postex[:, :, 1:, :] - postex[:, :, :-1, :]
            vcenterdu = geodu[:, :, 32::64, 32::64].permute(0, 2, 3, 1)
            vcenterdv = geodv[:, :, 32::64, 32::64].permute(0, 2, 3, 1)  # match v0 (channels last)

        elif self.nprims == 512:
            primpos = postex[:, :, 32::64, 16::32].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 16.0

        elif self.nprims == 4096:
            primpos = postex[:, :, 8::16, 8::16].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 32.0

        elif self.nprims == 16384:
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

            geodu = postex[:, :, :, 1:] - postex[:, :, :, :-1]
            geodv = postex[:, :, 1:, :] - postex[:, :, :-1, :]
            vcenterdu = geodu[:, :, 4::8, 4::8].permute(0, 2, 3, 1)
            vcenterdv = geodv[:, :, 4::8, 4::8].permute(0, 2, 3, 1)  # match v0 (channels last)

        elif self.nprims == 32768:
            primpos = postex[:, :, 4::8, 2::4].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 64.0

        elif self.nprims == 262144:
            primpos = postex[:, :, 1::2, 1::2].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 128.0

        else:
            raise ValueError(f"Unsupported number of primitives: {self.nprims}")

        if vcenterdu is None:
            raise ValueError(
                "u coordinate centres have not been computed yet, which means the requested number of primitives is not supported yet"
            )
        if vcenterdv is None:
            raise ValueError(
                "v coordinate centres have not been computed yet, which means the requested number of primitives is not supported yet"
            )

        # # compute TBN matrix
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

        if trainiter <= self.postrainstart:
            primposresid = primposresid * 0.0
            primrvecresid = primrvecresid * 0.0
            primscaleresid = primscaleresid * 0.0 + 1.0
        elif trainiter <= 2 * self.postrainstart:
            weight = (2 * self.postrainstart - trainiter) / (self.postrainstart + 1)
            primposresid = primposresid * (1 - weight) + weight * 0.0
            primrvecresid = primrvecresid * (1 - weight) + weight * 0.0
            primscaleresid = primscaleresid * (1 - weight) + weight * 1.0

        primpos = primpos + primposresid
        primrotresid = self.rodrig(primrvecresid.view(-1, 3)).view(encoding.size(0), nprims, 3, 3)
        primrot = torch.bmm(primrot.view(-1, 3, 3), primrotresid.view(-1, 3, 3)).view(encoding.size(0), nprims, 3, 3)
        primscale = primscale * primscaleresid

        viewdirs = viewpos / torch.sqrt(torch.sum(viewpos**2, dim=1, keepdim=True))

        z_tex, b_tex = id_cond["z_tex"], id_cond["b_tex"]
        primrgb = self.rgbdec(encoding, z_tex, b_tex, view=viewdirs)

        # TODO(julieta) this is denormalizing with hardcoded values... do something about this
        template = torch.cat([F.relu(primrgb * 25.0 + 100.0), F.relu(primalpha)], dim=2)

        losses = dict()
        if "primvolsum" in loss_set:
            losses["primvolsum"] = torch.sum(torch.prod(1.0 / primscale, dim=-1), dim=-1)

        return {
            "verts": geo,
            "template": template,
            "primpos": primpos,
            "primrot": primrot,
            "primscale": primscale,
        }, losses
