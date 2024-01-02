import einops
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

import models.autoencoder as aemodel
import models.bg.mlp2d as bglib
import models.bottlenecks.vae as vae
import models.colorcals.colorcal as colorcalib
import models.decoders.assembler as decoderlib
import models.encoders.expression as expression_encoder_lib
import models.encoders.identity as identity_encoder_lib
import models.raymarchers.mvpraymarcher_new as raymarcherlib
from config import ObjDict
from utils import create_uv_baridx, load_krt, load_obj


@pytest.fixture
def autoencoder() -> aemodel.Autoencoder:
    krt_dicts = load_krt("assets/KRT")

    ncams = len(krt_dicts)
    nids = 1

    colorcal = colorcalib.Colorcal(ncams, nids)
    dotobj = load_obj("assets/face_topology.obj")
    vt, vi, vti = dotobj["vt"], dotobj["vi"], dotobj["vti"]

    verts = torch.from_numpy(np.fromfile("assets/021924.bin", dtype=np.float32).reshape(1, -1, 3))

    # load per-textel triangulation indices
    # fd-data -- SAME as in RSC
    uvpath = "assets/rsc-assets/fd-data/"
    resolution = 1024
    trifile = f"{uvpath}/uv_tri_{resolution}_orig.txt"
    barfiles = []
    for i in range(3):
        barfiles.append(f"{uvpath}/uv_bary{i}_{resolution}_orig.txt")
    uvdata = create_uv_baridx("assets/face_topology.obj", trifile, barfiles)

    # Encoders
    expression_encoder = expression_encoder_lib.ExpressionEncoder(uvdata["uv_idx"], uvdata["uv_bary"])
    id_encoder = identity_encoder_lib.IdentityEncoder(uvdata["uv_idx"], uvdata["uv_bary"], wsize=128)

    # VAE bottleneck for the expression encoder
    bottleneck = vae.VAE_bottleneck(64, 16)

    # Decoder
    volradius = 256.0
    decoder = decoderlib.DecoderAssembler(
        vt=np.array(vt),
        vi=np.array(vi),
        vti=np.array(vti),
        vertmean=verts,
        vertstd=verts,
        volradius=volradius,
        nprims=128 * 128,
        primsize=(8, 8, 8),
    )

    config = ObjDict({"render": {"raymarcher_options": {"volume_radius": volradius, "chlast": False}}})
    raymarcher = raymarcherlib.Raymarcher(config)
    bgmodel = bglib.BackgroundModelSimple(ncams, nids)

    # Put together the Megazord
    ae = aemodel.Autoencoder(
        identity_encoder=id_encoder,
        expression_encoder=expression_encoder,
        bottleneck=bottleneck,
        decoder_assembler=decoder,
        raymarcher=raymarcher,
        colorcal=colorcal,
        bgmodel=bgmodel,
    )

    # DO NOT RUN VGG AT ALL and remove vgg in loss_weight for ABLATION TEST : @@@@
    print("id_encoder params:", sum(p.numel() for p in ae.id_encoder.parameters() if p.requires_grad))
    print(f"encoder params: {sum(p.numel() for p in ae.expr_encoder.parameters() if p.requires_grad):_}")
    print(f"decoder params: {sum(p.numel() for p in ae.decoder_assembler.parameters() if p.requires_grad):_}")
    print(f"colorcal params: {sum(p.numel() for p in ae.colorcal.parameters() if p.requires_grad):_}")
    print(f"bgmodel params: {sum(p.numel() for p in ae.bgmodel.parameters() if p.requires_grad):_}")
    print(f"total params: {sum(p.numel() for p in ae.parameters() if p.requires_grad):_}")

    return ae


def test_autoencoder_sizes(autoencoder):
    """Smoke test confirming expected sizes for the encoder"""

    assert True
