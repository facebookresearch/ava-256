import copy

import numpy as np
import torch

from models.decoders.assembler import DecoderAssembler
from utils import load_obj


def test_decoder_assembler_sizes():
    """Ensure the sizes of the rgb decoder make sense"""

    nprims = 128 * 128
    primsize = (8, 8, 8)

    # Load topology
    dotobj = load_obj("assets/face_topology.obj")
    vt, vi, vti = dotobj["vt"], dotobj["vi"], dotobj["vti"]

    # Load a dummy geometry object to get plausible mean and std verts
    verts = torch.from_numpy(np.fromfile("assets/021924.bin", dtype=np.float32).reshape(1, -1, 3))

    decoder = DecoderAssembler(
        vt=np.array(vt),
        vi=np.array(vi),
        vti=np.array(vti),
        vertmean=verts,
        vertstd=verts,
        volradius=256.0,
        nprims=nprims,
        primsize=primsize,
    )

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

    id_cond = {
        "z_tex": id_enc,
        "z_geo": id_enc,
        "b_tex": id_biases,
        "b_geo": copy.deepcopy(id_biases),
    }

    decouts = decoder(id_cond, ex_enc, view)

    assert decouts["verts"].shape == verts.shape
    assert decouts["template"].shape == torch.Size([1, 128**2, 4, *primsize])
    assert decouts["primpos"].shape == torch.Size([1, 128**2, 3])
    assert decouts["primrot"].shape == torch.Size([1, 128**2, 3, 3])
    assert decouts["primscale"].shape == torch.Size([1, 128**2, 3])
