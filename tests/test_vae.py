import torch

from models.bottlenecks.vae import VAE_bottleneck


def test_bottleneck_sizes():
    """Smoke test confirming expected sizes for the bottleneck"""

    bottleneck = VAE_bottleneck(64, 16)

    expr_code = torch.rand([1, 64, 4, 4])

    expr_code, mu, std = bottleneck(expr_code)

    assert expr_code.shape == torch.Size([1, 16, 4, 4])
    assert mu.shape == torch.Size([1, 16, 4, 4])
    assert std.shape == torch.Size([1, 16, 4, 4])
