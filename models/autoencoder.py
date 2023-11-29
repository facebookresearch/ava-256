"""
Volumetric autoencoder (image -> encoding -> volume -> image)
"""
from typing import Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from extensions.computeraydirs.computeraydirs import compute_raydirs


def color_normalize(src, dst):
    b, h, w = src.shape[0], src.shape[-2], src.shape[-1]
    A = src.view(-1, 3, w * h)
    B = dst.view(-1, 3, w * h)

    # mean normalize
    Amean = torch.mean(A, dim=-1, keepdim=True)
    Bmean = torch.mean(B, dim=-1, keepdim=True)
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
        *,
        vertmean: np.ndarray,
        vertstd: np.ndarray,
        texmean: np.ndarray,
        texstd: np.ndarray,
        imagemean: float,
        imagestd: float,
        identity_encoder: nn.Module,
        expression_encoder: nn.Module,
        bottleneck: nn.Module,
        decoder_assembler: nn.Module,
        raymarcher: nn.Module,
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

        # Register normalization stats
        # TODO(julieta) should (de)normalization live outside of the model?
        self.register_buffer("vertmean", torch.from_numpy(vertmean), persistent=False)
        self.vertstd = vertstd
        self.register_buffer("texmean", torch.from_numpy(texmean), persistent=False)
        self.texstd = texstd
        self.imagemean = imagemean
        self.imagestd = imagestd

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
        # Select pixels to evalaute ray marching on
        pixelcoords: torch.Tensor,
        # Indexing for background/color modeling
        idindex: Optional[torch.Tensor] = None,
        camindex: Optional[torch.Tensor] = None,
        # encoding: Optional[torch.Tensor] = None,
        id_cond: Optional[dict] = None,
        # image: Optional[torch.Tensor] = None,  # NOTE(julieta) the image is only needed for loss computation. Do that outside
        # imagemask: Optional[torch.Tensor] = None,
        bg: Optional[torch.Tensor] = None,
        # segmentation: Optional[torch.Tensor] = None,
        running_avg_scale: bool = False,
        gt_geo: Optional[torch.Tensor] = None,
        residuals_weight: float = 1.0,
        output_set: Set[str] = set(),
    ):
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
        id_cond = self.id_encoder(neut_verts, neut_avgtex)

        # Step 1. Get expression encoding
        expr_code = self.expr_encoder(
            verts=verts,
            avgtex=avgtex,
            neut_verts=neut_verts,
            neut_avgtex=neut_avgtex,
        )

        expr_code, expr_mu, expr_logstd = self.bottleneck(expr_code)

        # NOTE(julieta) expr_mu and expr_logstsd are used to compute kl divergence

        # compute relative viewing position
        # NOTE(julieta) should we be passing viewdir instead of pos to the decoder?
        # viewrot = torch.bmm(camrot, modelmatrix[:, :3, :3])
        viewpos = torch.bmm((campos[:, :] - modelmatrix[:, :3, 3])[:, None, :], modelmatrix[:, :3, :3])[:, 0, :]
        # viewdir = viewpos / torch.sqrt(torch.sum(viewpos**2, dim=-1, keepdim=True))

        decout = self.decoder_assembler(
            verts,
            id_cond,
            expr_code,
            viewpos,
            running_avg_scale,
            gt_geo,
            residuals_weight,
        )

        # TODO(julieta) check whether this makes sense
        samplecoords = torch.cat(
            [
                pixelcoords[..., :1] * 2 / (pixelcoords.shape[-2] - 1) - 1,
                pixelcoords[..., 1:] * 2 / (pixelcoords.shape[-3] - 1) - 1,
            ],
            dim=-1,
        )

        # Compute ray directions
        raypos, raydir, tminmax = compute_raydirs(
            campos, camrot, focal, princpt, pixelcoords, self.raymarcher.volume_radius
        )

        # Ray march across the ray directions to see the output image
        rayrgb, rayalpha, _, pos_img = self.raymarcher(
            raypos,
            raydir,
            tminmax,
            outputs=decout,
            with_pos_img=("pos_img" in output_set),
        )

        # color correction
        if (self.colorcal is not None) and (camindex is not None) and (idindex is not None):
            rayrgb = self.colorcal(rayrgb, camindex, idindex)

        # 4. Decode the background
        if bg is None and self.bgmodel is not None:
            # TODO(julieta) raise an error if bg is None and either camidx or idindex are None
            bg = self.bgmodel(camindex, idindex, samplecoords)

        # 6. matting
        if bg is not None:
            rayrgb = rayrgb + (1.0 - rayalpha) * bg
        else:
            # Add a black background
            black = [0, 0, 0]
            colour = np.asarray(black, dtype=np.float32)
            rayrgb = rayrgb + (1.0 - rayalpha) * torch.from_numpy(colour).to("cuda")[None, :, None, None]

        resultout = {
            # === Returned every time ===
            "encoding": expr_code,
            "expr_mu": expr_mu,
            "expr_logstd": expr_logstd,
            "irgbrec": rayrgb,
            # === Returned if asked for, useful for debugging and visualization ===
            "id_cond": id_cond if "idcond" in output_set else None,
            "samplecoords": samplecoords if "samplecooords" in output_set else None,
            "pos_img": pos_img if "pos_img" in output_set else None,
            "bg": bg if "bg" in output_set else None,
            "ialpha": rayalpha if "ialpha" in output_set else None,
        }
        return resultout
