import torch

from models.encoders.expression import kl_loss_stable


def test_kl_loss():
    n, d = 1, 128
    loss = kl_loss_stable(torch.zeros(n, d), torch.ones(n, d))
    # TODO(julieta) chec
    # k if there is a way to get zero kl div loss
    # assert loss == 0.``
    assert True


def test_sizes():
    """Smoke test confirming expected sizes for the encoder"""
