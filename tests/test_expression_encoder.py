import einops
import numpy as np
import torch
from PIL import Image

from models.encoders.expression import ExpressionEncoder
from utils import create_uv_baridx, load_obj, make_closest_uv_barys


def create_uv_baridx2(vt, vi, vti, uvtri, bar):
    """Julieta's port using the output of `make_closes_uv_barys` directly"""
    import cv2

    vt[:, 1] = 1 - vt[:, 1]  # note: flip y-axis

    idx0 = cv2.flip(vi[uvtri, 0], flipCode=0)
    idx1 = cv2.flip(vi[uvtri, 1], flipCode=0)
    idx2 = cv2.flip(vi[uvtri, 2], flipCode=0)
    bar0 = cv2.flip(bar[0], flipCode=0)
    bar1 = cv2.flip(bar[1], flipCode=0)
    bar2 = cv2.flip(bar[2], flipCode=0)

    return {
        "uv_idx": np.concatenate((idx0[None, :, :], idx1[None, :, :], idx2[None, :, :]), axis=0),
        "uv_bary": np.concatenate((bar0[None, :, :], bar1[None, :, :], bar2[None, :, :]), axis=0),
        "uv_coord": vt,
        "uv_tri": vti,
        "tri": vi,
    }


def test_encoder_sizes():
    """Smoke test confirming expected sizes for the encoder"""
    # TODO(julieta) make a centralized file with asset names so we don't have to type them out every time
    # TODO(julieta) have not been able to reproduce the bary_img

    # Load topology
    dotobj = load_obj("assets/face_topology.obj")
    vt, vi, vti = dotobj["vt"], dotobj["vi"], dotobj["vti"]

    # TODO(julieta) the above function takes ~40 seconds on my threadripper. We need to speed this up.
    uv_shape = 1024
    index_img, bary_img, _ = make_closest_uv_barys(
        torch.from_numpy(vt),
        torch.from_numpy(vti),
        uv_shape,
        # flip_uv=False,
    )
    bary_img = einops.rearrange(bary_img, "H W C -> C H W")

    # Load real verts, create dummy avg verts
    verts = torch.from_numpy(np.fromfile("assets/021924.bin", dtype=np.float32).reshape(1, -1, 3))
    neut_verts = verts + torch.rand(*verts.shape, dtype=torch.float32)

    # Load textures
    avgtex = np.array(Image.open("assets/021924.png")).astype(np.float32)
    avgtex = torch.from_numpy(einops.rearrange(avgtex, "H W C -> 1 C H W"))
    neut_avgtex = avgtex + torch.rand(*avgtex.shape, dtype=torch.float32)

    # fd-data -- SAME as in RSC
    uvpath = "assets/rsc-assets/fd-data/"
    resolution = 1024
    trifile = f"{uvpath}/uv_tri_{resolution}_orig.txt"
    barfiles = []
    for i in range(3):
        barfiles.append(f"{uvpath}/uv_bary{i}_{resolution}_orig.txt")
    uvdata = create_uv_baridx("assets/face_topology.obj", trifile, barfiles)

    # Created locally, is not the same unfortunately :(
    uvdata2 = create_uv_baridx2(vt, vi, vti, index_img.numpy(), bary_img.numpy())

    # Create expression encoder
    expression_encoder = ExpressionEncoder(uvdata["uv_idx"], uvdata["uv_bary"])
    expr_code = expression_encoder(verts, avgtex, neut_verts, neut_avgtex)

    assert expr_code.shape == torch.Size([1, 64, 4, 4])

    # Create expression encoder with data2
    expression_encoder = ExpressionEncoder(uvdata2["uv_idx"], uvdata2["uv_bary"])
    expr_code = expression_encoder(verts, avgtex, neut_verts, neut_avgtex)

    assert expr_code.shape == torch.Size([1, 64, 4, 4])
