# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Volumetric autoencoder (image -> encoding -> volume -> image)
"""

from typing import Dict, Optional, Set

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from extensions.utils.utils import compute_raydirs
from models.raymarchers.mvpraymarcher import Raymarcher


class Autoencoder(nn.Module):
    def __init__(
        self,
        *,
        # TODO(julieta) should we create more specific types for these submodules?
        identity_encoder: nn.Module,
        expression_encoder: nn.Module,
        bottleneck: nn.Module,
        decoder_assembler: nn.Module,
        raymarcher: Raymarcher,
        colorcal: Optional[nn.Module] = None,
        bgmodel: Optional[nn.Module] = None,
    ):
        """"""
        super(Autoencoder, self).__init__()

        self.id_encoder = identity_encoder
        self.expr_encoder = expression_encoder
        self.bottleneck = bottleneck
        self.decoder_assembler = decoder_assembler
        self.raymarcher = raymarcher
        self.colorcal = colorcal
        self.bgmodel = bgmodel

    # @profile
    def forward(
        self,
        # Camera parameters
        camrot: torch.Tensor,
        campos: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
        modelmatrix: torch.Tensor,
        # Encoder inputs,
        avgtex: torch.Tensor,
        verts: torch.Tensor,
        neut_avgtex: torch.Tensor,
        neut_verts: torch.Tensor,
        target_neut_avgtex: torch.Tensor,
        target_neut_verts: torch.Tensor,
        # Select pixels to evalaute ray marching on
        pixelcoords: torch.Tensor,
        # Indexing for background/color modeling
        idindex: Optional[torch.Tensor] = None,
        camindex: Optional[torch.Tensor] = None,
        # encoding: Optional[torch.Tensor] = None,
        id_cond: Optional[dict] = None,
        bg: Optional[torch.Tensor] = None,
        # segmentation: Optional[torch.Tensor] = None,
        running_avg_scale: bool = False,
        gt_geo: Optional[torch.Tensor] = None,
        residuals_weight: float = 1.0,
        output_set: Set[str] = set(),
        force_neutral=False,
        alpha_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Params
            camrot: [B, 3, 3] Rotation matrix of target view camera
            campos: [B, 3] Position of target view camera
            focal: [B, 2] Focal length of target view camera
            princpt: [B, 2] Princple point of target view camera
            modelmatrix: [B, 3, 3] Relative transform from the 'neutral' pose of object
            avgtex: [B, 3, 1024, 1024] Texture map averaged from all viewpoints
            verts: [B, 7306, 3] Mesh vertex positions

            camindex : torch.Tensor[int32], optional [B] Camera index within the list of all cameras
            pixelcoords : [B, H', W', 2] Pixel coordinates to render of the target view camera
            alpha_mask: [U, V] Alpha mask appied to the primitives in the UV space

            validinput : torch.Tensor, optional [B] Whether the current batch element is valid (used for missing images)
            fixedcamimage : torch.Tensor, optional [B, 3, 512, 334] Camera images from a one or more cameras that are always the same (i.e., unrelated to target)
            encoding : torch.Tensor, optional [B, 256] Direct encodings (overrides encoder)
            image : torch.Tensor, optional [B, 3, H, W] Target image
            imagemask : torch.Tensor, optional [B, 1, H, W] Target image mask for computing reconstruction loss
            imagevalid : torch.Tensor, optional [B]
            bg : torch.Tensor, optional [B, 3, H, W]
            renderoptions : dict Rendering/raymarching options (e.g., stepsize, whether to output debug images, etc.)
            outputlist : list What values to return (e.g., image reconstruction, debug output)

        Returns:
            Dictionary with items specified in output_set
        """

        resultout = dict()

        # get identity conditioning
        if (neut_verts is None) or (neut_avgtex is None):
            raise ValueError(f"Empty identity conditioning data")

        # Return {z_tex, b_tex, z_geo, b_geo}
        # Step 0. Get identity encoding
        if id_cond is None:
            id_cond = self.id_encoder(target_neut_verts, target_neut_avgtex)
        else:
            assert target_neut_avgtex is not None, "Must provide target_neut_avgtex or id_cond"
            assert target_neut_verts is not None, "Must provide target_neut_verts or id_cond"

        # Step 1. Get expression encoding
        expr_code = self.expr_encoder(
            verts=verts,
            avgtex=avgtex,
            neut_verts=neut_verts,
            neut_avgtex=neut_avgtex,
        )

        if force_neutral:
            expr_code = torch.zeros_like(expr_code)

        expr_code, expr_mu, expr_logstd = self.bottleneck(expr_code)

        # NOTE(julieta) expr_mu and expr_logstsd are used to compute kl divergence

        # Update encoder outputs
        encout = {
            # === Returned every time ===
            "encoding": expr_code,
            "expr_mu": expr_mu,
            "expr_logstd": expr_logstd,
            # === Returned if asked for, useful for debugging and visualization ===
            "id_cond": id_cond if "idcond" in output_set else None,
        }
        resultout.update(encout)

        # Step2: Decode and Render from expr_code and id_cond
        decout = self.decode(
            camrot=camrot,
            campos=campos,
            focal=focal,
            princpt=princpt,
            modelmatrix=modelmatrix,
            id_cond=id_cond,
            expr_encoding=expr_code,
            pixelcoords=pixelcoords,
            idindex=idindex,
            camindex=camindex,
            bg=bg,
            running_avg_scale=running_avg_scale,
            gt_geo=gt_geo,
            residuals_weight=residuals_weight,
            output_set=output_set,
            alpha_mask=alpha_mask,
        )

        # Update decoder outputs
        resultout.update(decout)

        return resultout

    def decode(
        self,
        # Camera parameters
        camrot: torch.Tensor,
        campos: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
        modelmatrix: torch.Tensor,
        # Encoder inputs: ID Encoder & Expr Encoder
        id_cond: dict,
        expr_encoding: torch.Tensor,
        # Select pixels to evalaute ray marching on
        pixelcoords: torch.Tensor,
        # Indexing for background/color modeling
        idindex: Optional[torch.Tensor] = None,
        camindex: Optional[torch.Tensor] = None,
        bg: Optional[torch.Tensor] = None,
        running_avg_scale: bool = False,
        gt_geo: Optional[torch.Tensor] = None,
        residuals_weight: float = 1.0,
        output_set: Set[str] = set(),
        alpha_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Method to only run the decoder and renderer
        """
        # compute relative viewing position
        # NOTE(julieta) should we be passing viewdir instead of pos to the decoder?
        # viewrot = torch.bmm(camrot, modelmatrix[:, :3, :3])
        viewpos = torch.bmm((campos[:, :] - modelmatrix[:, :3, 3])[:, None, :], modelmatrix[:, :3, :3])[:, 0, :]
        # viewdir = viewpos / torch.sqrt(torch.sum(viewpos**2, dim=-1, keepdim=True))

        # Decoding Step 1. Get decoder output
        decout = self.decoder_assembler(
            id_cond,
            expr_encoding,
            viewpos,
            running_avg_scale,
            gt_geo,
            residuals_weight,
        )

        # Apply alpha mask to the primitives
        if alpha_mask is not None:
            nh = int(np.sqrt(self.decoder_assembler.nprims))
            alpha_mask_strided = F.interpolate(alpha_mask.unsqueeze(0).unsqueeze(0), size=(nh, nh), antialias=False)[0][
                0
            ]
            valid_prims = alpha_mask_strided.reshape(-1).bool()
            assert (
                valid_prims.shape[0] == decout["template"].shape[1]
            ), f"valid_prims: {valid_prims.shape}, template: {decout['template'].shape}"
            decout["template"] = decout["template"][:, valid_prims].contiguous()
            decout["primpos"] = decout["primpos"][:, valid_prims].contiguous()
            decout["primrot"] = decout["primrot"][:, valid_prims].contiguous()
            decout["primscale"] = decout["primscale"][:, valid_prims].contiguous()
            if "warp" in decout:
                decout["warp"] = decout["warp"][:, valid_prims].contiguous()

        # TODO(julieta) check whether this makes sense
        samplecoords = torch.cat(
            [
                pixelcoords[..., :1] * 2 / (pixelcoords.shape[-2] - 1) - 1,
                pixelcoords[..., 1:] * 2 / (pixelcoords.shape[-3] - 1) - 1,
            ],
            dim=-1,
        )

        # Compute ray directions for ray marching
        raypos, raydir, tminmax = compute_raydirs(
            campos, camrot, focal, princpt, pixelcoords, self.raymarcher.volume_radius
        )

        # Decoding Step 2. Get decoder output
        # Ray march across the ray directions to see the output image
        rayrgb, rayalpha, _, pos_img = self.raymarcher(
            raypos,
            raydir,
            tminmax,
            decout,
            with_pos_img=("pos_img" in output_set),
        )

        # [Optional] Decoding Step 3. color correction
        if (self.colorcal is not None) and (camindex is not None) and (idindex is not None):
            rayrgb = self.colorcal(rayrgb, camindex, idindex)

        # [Optional] Decoding Step 4. Decode the background
        if bg is None and (self.bgmodel is not None and camindex is not None and idindex is not None):
            # TODO(julieta) raise an error if bg is None and either camidx or idindex are None
            bg = self.bgmodel(camindex, idindex, samplecoords)

        # Decoding Step 5. matting
        if bg is not None:
            rayrgb = rayrgb + (1.0 - rayalpha) * bg
        else:
            # Add a black background
            black = [0, 0, 0]
            colour = np.asarray(black, dtype=np.float32)
            rayrgb = rayrgb + (1.0 - rayalpha) * torch.from_numpy(colour).to("cuda")[None, :, None, None]

        decout = {
            # === Returned every time ===
            "irgbrec": rayrgb,
            "verts": decout["verts"],
            # === Returned if asked for, useful for debugging and visualization ===
            "primscale": decout["primscale"] if "primscale" in output_set else None,
            "samplecoords": samplecoords if "samplecooords" in output_set else None,
            "pos_img": pos_img if "pos_img" in output_set else None,
            "bg": bg if "bg" in output_set else None,
            "ialpha": rayalpha if "ialpha" in output_set else None,
        }

        return decout
