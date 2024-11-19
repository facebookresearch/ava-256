# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Universal facial encoder"""

from typing import Dict, List, Optional, Union

import timm
import torch
import torch.nn.functional as F
from einops import rearrange


class UniversalEncoder(torch.nn.Module):
    """
    An universal headset encoder that takes in a face and an eye image, optinally conditioning frames
    and encode them into a single vector.
    """

    def __init__(self, in_chans: int, out_chans: int, num_views: int, num_conditions: int = 1) -> None:
        super().__init__()
        self.num_views: int = num_views
        self.num_conditions: int = num_conditions

        # Create two separate models for eyes and faces
        eye_model = timm.create_model("tf_mobilenetv3_large_100", in_chans=in_chans)
        face_model = timm.create_model("tf_mobilenetv3_large_100", in_chans=in_chans)
        self.num_eye_features = eye_model.feature_info
        self.num_face_features = face_model.feature_info
        eye_model = eye_model.as_sequential()
        face_model = face_model.as_sequential()
        self.eye_model_precond, self.eye_model_postcond = eye_model[:7], eye_model[7:9]
        self.face_model_precond, self.face_model_postcond = face_model[:7], face_model[7:9]

        # Fusion module for current and conditional frames
        if num_conditions > 0:
            self.eye_fusion = torch.nn.Conv2d(
                self.num_eye_features[-2]["num_chs"] * (num_conditions + 1),
                self.num_eye_features[-2]["num_chs"],
                kernel_size=1,
                bias=False,
            )
            self.face_fusion = torch.nn.Conv2d(
                self.num_face_features[-2]["num_chs"] * (num_conditions + 1),
                self.num_face_features[-2]["num_chs"],
                kernel_size=1,
                bias=False,
            )

        # Regressor head, fusing multiple headset views' features together
        self.head = torch.nn.Sequential(
            torch.nn.Linear(
                (num_views // 2) * (self.num_eye_features[-1]["num_chs"] + self.num_face_features[-1]["num_chs"]), 512
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(512, out_chans),
        )

    def forward(
        self, x: torch.Tensor, x_cond: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Split the input into two parts (eye and face images), then run them through their respective models

        Args:
            x: Input images of shape [N, num_views, in_chans, H, W]. We assume eye/face images have an equal
            number of views.
            x_cond: Conditioning Input images of shape [N, num_views, num_conds, in_chans, H, W].

        Returns:
            A dictionary containing the expression code of shape [N, out_chans]
        """
        # Step 1. Feature extraction for eyes and faces
        # Process current frame
        eye_input, face_input = torch.chunk(x, 2, dim=1)
        # [Batch*Views, C, feat_H, feat_W]
        eye_features = self.eye_model_precond(eye_input.flatten(0, 1))
        face_features = self.face_model_precond(face_input.flatten(0, 1))

        # Process conditionings
        num_conds = self.num_conditions
        if num_conds > 0:
            eye_cond, face_cond = torch.chunk(x_cond, 2, dim=1)
            assert (
                eye_cond.shape[2] == self.num_conditions
            ), f"Conditioning shape mismatch! Got {eye_cond.shape}, expect {self.num_conditions}"
            eye_cond_features_pre = self.eye_model_precond(eye_cond.flatten(0, 2))
            face_cond_features_pre = self.face_model_precond(face_cond.flatten(0, 2))
            # [Batch*Views, Conds * C, feat_H, feat_W]
            eye_cond_features_pre = eye_cond_features_pre.unflatten(0, (-1, num_conds)).flatten(1, 2)
            face_cond_features_pre = face_cond_features_pre.unflatten(0, (-1, num_conds)).flatten(1, 2)

            # Step 2. Conditioning Fusion
            eye_features = torch.cat([eye_features, eye_cond_features_pre], dim=1)
            face_features = torch.cat([face_features, face_cond_features_pre], dim=1)
            # [Batch*Views, C, feat_H, feat_W]
            eye_features = self.eye_fusion(eye_features)
            face_features = self.face_fusion(face_features)
        else:
            assert x_cond is None, "Conditioning is set to 0, but got conditioning input!"

        # Step 3. Post cond feature extraction
        eye_features = F.adaptive_avg_pool2d(self.eye_model_postcond(eye_features), output_size=1).squeeze(-2, -1)
        eye_features = rearrange(eye_features, "(n v) c -> n (v c)", v=(self.num_views // 2))
        face_features = F.adaptive_avg_pool2d(self.face_model_postcond(face_features), output_size=1).squeeze(-2, -1)
        face_features = rearrange(face_features, "(n v) c -> n (v c)", v=(self.num_views // 2))

        # Step 4. Regressor head
        expr_code = self.head(torch.cat([eye_features, face_features], dim=-1))

        return {
            "expression": expr_code,
        }
