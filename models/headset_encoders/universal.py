# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Universal facial encoder"""

from typing import List, Dict, Tuple, Union

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class UniversalEncoder(torch.nn.Module):
    """
    A really simple universal headset encoder that takes in a face and an eye image, and encode
    them into a single vector.
    """
    def __init__(self, in_chans: int, out_chans: int, num_views: int):
        super().__init__()
        eye_model = timm.create_model("tf_mobilenetv3_large_100", in_chans=in_chans)
        face_model = timm.create_model("tf_mobilenetv3_large_100", in_chans=in_chans)
        self.num_eye_features = eye_model.feature_info
        self.num_face_features = face_model.feature_info
        eye_model = eye_model.as_sequential()
        face_model = face_model.as_sequential()
        self.eye_model_precond, self.eye_model_postcond = eye_model[:7], eye_model[7:9]
        self.face_model_precond, self.face_model_postcond = face_model[:7], face_model[7:9]

        # Fusion module for current and conditional frames
        self.fusion = torch.nn.Conv2d(
            self.num_eye_features[-2]["num_chs"] * 2,
            self.num_eye_features[-2]["num_chs"],
            kernel_size=1,
            bias=False
        )

        # Regressor head, fusing multiple headset views' features together
        self.head = torch.nn.Sequential(
            torch.nn.Linear(
                (num_views//2) * (self.num_eye_features[-1]["num_chs"] + self.num_face_features[-1]["num_chs"]),
                512
            ),
            torch.nn.SiLUI(),
            torch.nn.Linear(512, out_chans)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Split the input into two parts (eye and face images), then run them through their respective models

        Args:
            x: Input images of shape [N, num_views, in_chans, H, W]. We assume eye/face images have an equal
            number of views.
        """
        # Process eyes
        eye_input, face_input = torch.chunk(x, 2, dim=1)
        eye_features = F.adaptive_avg_pool2d(
            self.eye_model.forward_features(eye_input.flatten(0,1)), output_size=1
        ).squeeze(-2,-1)
        eye_features = rearrange(eye_features, "(n v) c -> n (v c)", v=(self.num_views // 2))

        # Process faces
        face_features = F.adaptive_avg_pool2d(
            self.face_model.forward_features(face_input.flatten(0,1)), output_size=1
        ).squeeze(-2,-1)
        face_features = rearrange(face_features, "(n v) c -> n (v c)", v=(self.num_views // 2))

        return {
            "expression": self.head(torch.cat([eye_features, face_features], dim=-1)),
        }
