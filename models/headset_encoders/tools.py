from typing import List, Dict, Set, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn


def rvec_to_R(rvec: torch.Tensor) -> torch.Tensor:
    """Computes the rotation matrix R from a tensor of Rodrigues vectors.

    n = ||rvec||
    rn = rvec/||rvec||
    N = [rn]_x = [[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]]
    R = I + sin(n)*N + (1-cos(n))*N*N
    """
    n = rvec.norm(dim=-1, p=2).clamp(min=1e-6)[..., None, None]
    rn = rvec / n[..., :, 0]
    zero = torch.zeros_like(n[..., 0, 0])
    N = torch.stack(
        (
            zero,
            -rn[..., 2],
            rn[..., 1],
            rn[..., 2],
            zero,
            -rn[..., 0],
            -rn[..., 1],
            rn[..., 0],
            zero,
        ),
        -1,
    ).view(rvec.shape[:-1] + (3, 3))
    R = (
        torch.eye(3, dtype=n.dtype, device=n.device).view([1] * (rvec.dim() - 1) + [3, 3])
        + torch.sin(n) * N
        + ((1 - torch.cos(n)) * N) @ N
    )
    return R


class PatchMixAugmentation:
    def __init__(self, aug_prob: float = 0.5, num_holes: int = 5, size: int = 48) -> None:
        """
        A CutMix (https://arxiv.org/abs/1905.04899) alike augmentation strategy that mixes the patches of
        different images together. Unlike `torchvision.transforms.v2.CutMix`, this does not require
        label information. Note: this applies on a batch of images.

        Args:
            aug_prob (float): probability of applying augmentation. Default is 0.5
            num_holes (int): number of holes to be swapped in the minibatch. Default is 1
            size (int): maximum size of hole to fill in the image. Default is 48
        """
        self.aug_prob = aug_prob
        self.num_holes = num_holes
        self.size = int(size)

    def corrupt(self, feature: torch.Tensor, pfeature: torch.Tensor) -> torch.Tensor:
        bsz, num_views, _, height, width = feature.shape
        device = feature.device
        g = torch.stack(
            [
                torch.arange(0, width, device=device)[None, ...].repeat(height, 1),
                torch.arange(0, height, device=device)[..., None].repeat(1, width),
            ],
            dim=2,
        ).float()
        h = (
            torch.stack(
                [
                    torch.randint(
                        low=0, high=width, size=(bsz, num_views, self.num_holes), device=device
                    ),
                    torch.randint(
                        low=0, high=height, size=(bsz, num_views, self.num_holes), device=device
                    ),
                ],
                dim=-1,
            )
            .float()
            .view(bsz, num_views, 1, self.num_holes, 1, 1, 2)
        )
        p = torch.rand(bsz, num_views, 1, 1, 1, device=device) < self.aug_prob
        d = ((g.view(1, 1, 1, height, width, 2) - h).abs() < (self.size // 2)).float().prod(dim=-1).sum(dim=3).clip(max=1.0)
        return torch.lerp(feature, pfeature, d * p)

    def augment(self, clean_feature: torch.Tensor) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        if self.aug_prob == 0.0:
            return clean_feature, None
        bsz = clean_feature.shape[0]
        perm = torch.randperm(bsz, device=clean_feature.device)
        p_clean_feature = clean_feature[perm]
        corr_feature = self.corrupt(clean_feature, p_clean_feature)
        return corr_feature, perm
