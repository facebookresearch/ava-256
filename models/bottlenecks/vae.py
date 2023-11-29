# variational autoencoder bottleneck

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import models.utils


def kl_loss_stable(mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    """Stable version of kl divergence"""
    return torch.mean(-0.5 + torch.abs(logstd) + 0.5 * mu**2 + 0.5 * torch.exp(2 * -1 * torch.abs(logstd)), dim=-1)


class VAE_bottleneck(nn.Module):
    """Compresses a feature map and trains it as a VAE"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mean_squash: Optional[float] = 0.1,
        std_squash: Optional[float] = 0.01,
    ):
        super(VAE_bottleneck, self).__init__()

        c = models.utils.Conv2dWN

        self.mu = c(in_dim, out_dim, 1, 1, 0)
        self.logstd = c(in_dim, out_dim, 1, 1, 0)
        self.mean_squash = mean_squash
        self.std_squash = std_squash

        models.utils.initmod(self.mu)
        models.utils.initmod(self.logstd)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Squash
        mu = self.mu(x) * self.mean_squash
        logstd = self.logstd(x) * self.std_squash

        # Sample if training
        if self.training:
            z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)
        else:
            z = mu

        return z, mu, logstd
