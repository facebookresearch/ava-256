from math import sqrt
from typing import Dict, Optional, Tuple

import einops
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
        idxim: np.ndarray,
        barim: np.ndarray,
        vertmean: torch.Tensor,
        vertstd: torch.Tensor,
        volradius: float,
        nprims: int = 128 * 128,
        primsize: Tuple[int, int, int] = (8, 8, 8),
    ):
        super(DecoderAssembler, self).__init__()

        self.volradius = volradius

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

        self.register_buffer("vertmean", vertmean, persistent=False)
        self.vertstd = vertstd

        idxim = einops.rearrange(idxim, "C H W -> H W C")
        barim = einops.rearrange(barim, "C H W -> H W C")
        self.register_buffer("idxim", torch.tensor(idxim).long(), persistent=False)
        self.register_buffer("barim", torch.tensor(barim), persistent=False)

        self.register_buffer("adaptwarps", torch.zeros(self.nprims))

    def forward(
        self,
        id_cond: Dict[str, torch.Tensor],
        expr_encoding: torch.Tensor,
        viewpos: torch.Tensor,
        running_avg_scale: bool = False,
        gt_geo: Optional[torch.Tensor] = None,
        residuals_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Params:
            encoding : [B, 256] Expression encoding of current frame
            id_cond: Dictionary with geometry and texture identity biases and code
            viewpos: [B, 3] Viewing position of target camera view
            running_avg_scale: Whether to compute the scale with a running average
            gt_geo: [B, 7306, 3] Ground truth geometry. If passed, this is used as a guide mesh
            residuals_weight: Float deciding how much the residuals should be taken into account, as opposed to:
                zero for positions, zero for rotations (in rodrigues' representations zero vec is an identity rotation),
                and one for scale
        Returns
            A first dictionary contains predicted geometry (vertex positions), primitive contents (rgb and alpha) as
            well as locations, scaling, and rotations.
        """
        nprims = self.nprims

        z_geo, b_geo = id_cond["z_geo"], id_cond["b_geo"]
        (
            primalpha,
            geo,
            primitives_position_residuals,
            primitives_rotation_residuals,
            primitives_scale_residuals,
        ) = self.geodec(expr_encoding, z_geo, b_geo)
        geo = geo * self.vertstd + self.vertmean

        predicted_geo = geo

        if gt_geo is not None:
            # NOTE(julieta) If ground truth geometry is passed, use it as guiding mesh for the primitive placement.
            # This is useful at the beginning of training, when the predicted geometry is bad. We still return the
            # predicted geometry, so the geo branch is still getting trained
            geo = gt_geo * self.vertstd + self.vertmean

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
            if running_avg_scale:
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
            if running_avg_scale:
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

        # Compute TBN matrix
        tangent = vcenterdu
        tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-8)
        normal = torch.cross(tangent, vcenterdv, dim=3)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True).clamp(min=1e-8)
        bitangent = torch.cross(normal, tangent, dim=3)
        bitangent = bitangent / torch.norm(bitangent, dim=-1, keepdim=True).clamp(min=1e-8)
        primrot = (
            torch.stack((tangent, bitangent, normal), dim=-2)
            .view(expr_encoding.size(0), -1, 3, 3)
            .contiguous()
            .permute(0, 1, 3, 2)
            .contiguous()
        )

        rw = sorted([0.0, residuals_weight, 1.0])[1]  # clamp between 0 and 1
        if rw < 1.0:
            primitives_position_residuals = primitives_position_residuals * rw
            primitives_rotation_residuals = primitives_rotation_residuals * rw
            primitives_scale_residuals = primitives_scale_residuals * rw + (1 - rw)

        primpos = primpos + primitives_position_residuals
        primrotresid = self.rodrig(primitives_rotation_residuals.view(-1, 3)).view(expr_encoding.size(0), nprims, 3, 3)
        primrot = torch.bmm(primrot.view(-1, 3, 3), primrotresid.view(-1, 3, 3)).view(
            expr_encoding.size(0), nprims, 3, 3
        )
        primscale = primscale * primitives_scale_residuals

        viewdirs = viewpos / torch.sqrt(torch.sum(viewpos**2, dim=1, keepdim=True))

        z_tex, b_tex = id_cond["z_tex"], id_cond["b_tex"]
        primrgb = self.rgbdec(expr_encoding, z_tex, b_tex, view=viewdirs)

        # TODO(julieta) this is denormalizing with hardcoded values... do something about this
        template = torch.cat([F.relu(primrgb * 25.0 + 100.0), F.relu(primalpha)], dim=-1)

        return {
            "verts": predicted_geo,
            "template": template,
            "primpos": primpos,
            "primrot": primrot,
            "primscale": primscale,
        }
