"""Image-based losses"""

import torch


def mean_ell_1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    diff = pred - gt
    return diff.abs().mean()


def mean_ell_2(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    diff = pred - gt
    return (diff**2).mean()


# TODO(julieta) add ssim?
