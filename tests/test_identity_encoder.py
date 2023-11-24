import torch

from models.encoders.identity import EncoderUNet


def test_unet_encoder_sizes():
    """Smoke test confirming expected sizes for the encoder"""

    device = "cpu"
    img = torch.rand((1, 3, 1024, 1024), dtype=torch.float32).to(device)
    unet_encoder = EncoderUNet().to(device)
    with torch.no_grad():
        z, biases = unet_encoder(img)

    # Smallest id output is of the same shape as the id encoder
    assert z.shape == torch.Size([1, 16, 4, 4])

    # The biases come from smaller to larger, shapes are 8 -> 1024 in powers of two
    # fmt: off
    bias_channels = [256, 128, 128, 64,  64,  32,  16,    3]
    bias_shapes =   [8,    16,  32, 64, 128, 256, 512, 1024]
    for bias, c, h in zip(biases, bias_channels, bias_shapes):
        assert bias.shape == torch.Size([1, c, h, h])
    # fmt: on
