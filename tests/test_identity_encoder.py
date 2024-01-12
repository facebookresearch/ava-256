import einops
import numpy as np
import pytest
import torch
from PIL import Image

from models.encoders.identity import GeoTexCombiner, IdentityEncoder, UnetEncoder
from utils import create_uv_baridx


@pytest.fixture
def unet_encoder() -> UnetEncoder:
    return UnetEncoder()


@pytest.fixture
def img() -> torch.Tensor:
    return torch.rand((1, 3, 1024, 1024), dtype=torch.float32)


def test_unet_encoder_sizes(unet_encoder, img):
    """Smoke test confirming expected sizes for the encoder"""

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


def test_combiner_sizes(unet_encoder, img):
    """Ensure the combiner output the correct sizes"""

    combiner = GeoTexCombiner()
    with torch.no_grad():
        _, b_geo = unet_encoder(img)
        _, b_tex = unet_encoder(img)
        b_geo_combined, b_tex_combined = combiner(b_geo, b_tex)

    # The biases come from smaller to larger, shapes are 8 -> 1024 in powers of two
    # fmt: off
    bias_channels = [256, 128, 128, 64,  64,  32,  16,    3]
    bias_shapes =   [8,    16,  32, 64, 128, 256, 512, 1024]
    for bias_geo, bias_tex, c, h in zip(b_geo_combined, b_tex_combined, bias_channels, bias_shapes):
        assert bias_geo.shape == torch.Size([1, c, h, h])
        assert bias_tex.shape == torch.Size([1, c, h, h])
    # fmt: on


def test_identity_encoder_sizes():
    """Check the identity encoder produces the right sizes"""

    # Load real verts and an average texture
    verts = torch.from_numpy(np.fromfile("assets/021924.bin", dtype=np.float32).reshape(1, -1, 3))
    avgtex = np.array(Image.open("assets/021924_avgtex.png")).astype(np.float32)
    avgtex = torch.from_numpy(einops.rearrange(avgtex, "h w c -> 1 c h w"))

    # Create identity encoder
    uvdata = create_uv_baridx("assets/face_topology.obj", resolution=1024)
    identity_encoder = IdentityEncoder(uvdata["uv_idx"], uvdata["uv_bary"], wsize=128)
    with torch.no_grad():
        encoder_outs = identity_encoder(verts, avgtex)

    assert encoder_outs["z_geo"].shape == torch.Size([1, 16, 4, 4])
    assert encoder_outs["z_tex"].shape == torch.Size([1, 16, 4, 4])

    # fmt: off
    bias_channels = [256, 128, 128, 64,  64,  32,  16,    3]
    bias_shapes =   [8,    16,  32, 64, 128, 256, 512, 1024]
    for bias_geo, bias_tex, c, h in zip(encoder_outs["b_geo"], encoder_outs["b_tex"], bias_channels, bias_shapes):
        assert bias_geo.shape == torch.Size([1, c, h, h])
        assert bias_tex.shape == torch.Size([1, c, h, h])
    # fmt: on
