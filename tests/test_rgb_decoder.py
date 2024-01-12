import numpy as np
import torch

from models.decoders.rgb import RGBDecoder


def test_rgb_decoder_sizes():
    """Ensure the sizes of the rgb decoder make sense"""

    nprims = 128 * 128
    primsize = (8, 8, 8)
    imsize = int(np.sqrt(nprims)) * primsize[1]
    outch = 3
    decoder = RGBDecoder(imsize=imsize, nboxes=nprims, boxsize=primsize[0], outch=outch, viewcond=True)

    ex_enc = torch.rand([1, 16, 4, 4])
    id_enc = torch.rand([1, 16, 4, 4])
    id_biases = []

    # fmt: off
    bias_channels = [256, 128, 128, 64,  64,  32,  16,    3]
    bias_shapes =   [8,    16,  32, 64, 128, 256, 512, 1024]
    for c, h in zip(bias_channels, bias_shapes):
        id_biases.append(torch.rand([1, c, h, h]))
    # fmt: on

    view = torch.rand(1, 3)
    decout = decoder(ex_enc, id_enc, id_biases, view)

    assert decout.shape == torch.Size([1, nprims, *primsize, outch])
