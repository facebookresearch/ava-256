# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A wrapper for universal facial encoder"""

from typing import Any, List, Dict, Tuple, Union

from einops import rearrange
import numpy as np
import cv2
import imageio
import torch
import torch.nn.functional as F
from models.headset_encoders.ud import UDWrapper, FixedFrontalViewGenerator
from models.headset_encoders.tools import get_color_map


class UniversalEncoderLoss(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, identities: List[str]) -> None:
        """
        A wrapper for running, decoding the evaluating universal headset facial encoders

        Args:
            encoder: The universal decoder to use.
            identities: The list of decoder identity names to use.
        """
        super().__init__()
        self._encoder = encoder
        self._decoder = UDWrapper(
            ud_exp_name="/uca/leochli/oss/ava256_universal_decoder",
            identities=identities
        )
        self._view_generator = FixedFrontalViewGenerator(down_sample_factor=2)
        device = torch.device("cuda")
        self._decoder.to(device)
        self._decoder.requires_grad_(False)  # Do not update decoder parameters
        self._view_generator.to(device)

        self.ident_str_mapping = {ident[:6]: ident for ident in identities}
        face_mask = imageio.imread("assets/face_expression_mask.png")
        self.register_buffer("face_mask", torch.from_numpy(face_mask) / 255., persistent=False)
        face_weight_geo = np.load("assets/face_weight_geo.npy")
        self.register_buffer("face_weight_geo", torch.from_numpy(face_weight_geo), persistent=False)
        self.register_buffer("color_map", torch.from_numpy(get_color_map()).byte(), persistent=True)

    @property
    def encoder(self) -> torch.nn.Module:
        return self._encoder
    
    def crop_face_regions(self, diag: torch.Tensor) -> torch.Tensor:
        """
        Keep only the face region from the input image.

        Args:
            diag: The input image of shape [N, 3, H, W]
        """
        H, W = diag.shape[2:]
        face_H = slice(0, int(0.75 * H))
        face_W = slice(int(0.2 * W), -1)
        face_diag = diag[:, :, face_H, face_W]
        return face_diag

    def get_visual(self, outputs) -> np.ndarray:
        """
        Visualize the outputs of a forward pass.

        Args:
            outputs: The output dictionary from `forward`
        """
        with torch.no_grad():
            # Encoder visualization
            # [N, num_views, in_chans, H, W]
            headset_imgs_size = outputs["headset_cam_img"].shape[-1]
            encoder_diag = (outputs["headset_cam_img"] / 2 + 0.5).clamp(0, 1) * 255
            encoder_diag = rearrange(encoder_diag, "b v c h w -> b c h (v w)").clamp(0, 255).byte()

            # Decoder visualization
            diff = ((outputs["img_gt"] - outputs["img_pred"]) * outputs["mask"]).abs().mean(1)
            diff = (diff * 5).clamp(0, 255).long()
            diff = self.color_map[diff].permute(0, 3, 1, 2).float()
            img_gt = self.crop_face_regions(outputs["img_gt"])
            img_pred = self.crop_face_regions(outputs["img_pred"])
            diff = self.crop_face_regions(diff)
            decoder_diag = torch.cat([img_gt, img_pred, diff], dim=-1)
            decoder_diag = decoder_diag.clamp(0, 255).byte()  # (N, 3, H, W)

            # Merge encoder & decoder visualizations
            assert encoder_diag.shape[0] == decoder_diag.shape[0], "Batch size mismatch!"
            delta_H = decoder_diag.shape[2] - encoder_diag.shape[2]
            if delta_H > 0:
                encoder_diag = F.pad(encoder_diag, (0, 0, 0, delta_H))
            elif delta_H < 0:
                decoder_diag = F.pad(decoder_diag, (0, delta_H))
            diag_img = torch.cat([encoder_diag.expand(-1, 3, -1, -1), decoder_diag], dim=-1)

            # Permute and add title bar
            diag_img = diag_img.permute(0, 2, 3, 1).cpu()  # (N, H, W, 3)
            batch_size = diag_img.shape[0]
            font_size = 1
            title_text_args = (cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
            text_args = (cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
            title_bar = np.zeros((batch_size, 120, diag_img.shape[-2], 3), dtype=np.uint8)
            for i in range(batch_size):
                ident_str = outputs['index']['ident'][i]
                frame = outputs['index']['frame'][i]
                segment = outputs['index']['segment'][i]
                cv2.putText(title_bar[i], f"Identity: {ident_str}", (10, 30 * font_size), *title_text_args)
                cv2.putText(title_bar[i], f"frame: {frame}, segment: {segment}", (10, 60 * font_size), *title_text_args)
                cv2.putText(title_bar[i], "Headset images", (10, 110 * font_size), *text_args)
                cv2.putText(title_bar[i], "Groundtruth", (10+headset_imgs_size*4, 110 * font_size), *text_args)
                cv2.putText(
                    title_bar[i],
                    "Prediction",
                    (10+headset_imgs_size*4+img_gt.shape[-1], 110 * font_size),
                    *text_args
                )
                cv2.putText(
                    title_bar[i],
                    "Error",
                    (10+headset_imgs_size*4+img_gt.shape[-1]+img_pred.shape[-1], 110 * font_size),
                    *text_args
                )
            diag_img = torch.cat([torch.tensor(title_bar).byte(), diag_img], dim=1)
        return diag_img.flatten(0, 1).detach().cpu().numpy()

    def get_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        Compute the losses for universal facial encoders, given the forward pass outputs. Loss
        types with zero weight are skipped.

        Args:
            outputs: The output dictionary from `forward`
            loss_weights: The weighting for each type of loss
        """
        losses = {}
        if loss_weights.get("img_L1", 0) > 0:
            losses["img_L1"] = F.l1_loss(outputs["img_pred"], outputs["img_gt"], reduction="none") * outputs["mask"]
            losses["img_L1"] = losses["img_L1"].mean((-1, -2, -3)) / outputs["mask"].mean((-1, -2, -3))
        if loss_weights.get("expression", 0) > 0:
            losses["expression"] = F.mse_loss(outputs["expression"], outputs["expression_gt"], reduction="none")
            losses["expression"] = losses["expression"].mean((-1))
        if loss_weights.get("geo_weighted_L1", 0) > 0:
            losses["geo_weighted_L1"] = F.smooth_l1_loss(
                outputs["geo_pred"] * self.face_weight_geo[:, None],
                outputs["geo_gt"] * self.face_weight_geo[:, None],
                reduction="none"
            )
            losses["geo_weighted_L1"] = losses["geo_weighted_L1"].mean((-1, -2))
        return losses

    def forward(
        self, 
        data: Dict[str, Any], 
        loss_weights: Dict[str, float],
        use_face_mask: bool = True
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """Compute the latent expression code, and then decode"""
        outputs = {
            "headset_cam_img": data["headset_cam_img"], 
            "expression_gt": data["gt_latent_code"], 
            "index": data["index"]
        }
        identities = [self.ident_str_mapping[ident[:6]] for ident in data["index"]["ident"]]

        # Compute encoding
        x = data["headset_cam_img"]
        x_cond = data["cond_headset_cam_img"]
        bsz = x.shape[0]
        enc_outputs = self._encoder(x, x_cond)
        outputs.update(enc_outputs)

        # Compute decoding
        camera_params = self._view_generator(bsz, randomize=False)
        decoder_inputs = {
            "idents": identities,
            "camera_params": camera_params,
            "alpha_mask": self.face_mask if use_face_mask else None,
        }
        dec_outputs_pred = self._decoder.render({**decoder_inputs, "encodings": outputs["expression"]})
        with torch.no_grad():
            dec_outputs_gt = self._decoder.render({**decoder_inputs, "encodings": outputs["expression_gt"]})

        outputs["img_pred"] = dec_outputs_pred["img_pred"]
        outputs["geo_pred"] = dec_outputs_pred["verts"]
        outputs["img_gt"] = dec_outputs_gt["img_pred"]
        outputs["geo_gt"] = dec_outputs_gt["verts"]
        outputs["mask"] = ((dec_outputs_pred["mask"] > 0.999) & (dec_outputs_gt["mask"] > 0.999)).float()
        losses = self.get_losses(outputs, loss_weights)
        return outputs, losses
