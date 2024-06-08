# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Image-based losses"""

import torch


def mean_ell_1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    diff = pred - gt
    return diff.abs().mean()


def mean_ell_2(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    diff = pred - gt
    return (diff**2).mean()


# TODO(julieta) add ssim?
