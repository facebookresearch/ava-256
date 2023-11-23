import torch

from models.encoders.expression import ExpressionEncoder, kl_loss_stable
from utils import load_obj, make_closest_uv_barys


def test_kl_loss():
    n, d = 1, 128
    loss = kl_loss_stable(torch.zeros(n, d), torch.ones(n, d))
    # TODO(julieta) chec
    # k if there is a way to get zero kl div loss
    # assert loss == 0.``


def create_uv_baridx(geofile, trifile, barfiles):
    import cv2

    dotobj = load_obj(geofile, extension="obj")
    vt, vi, vti = dotobj["vt"], dotobj["vi"], dotobj["vti"]

    vt[:, 1] = 1 - vt[:, 1]  # note: flip y-axis
    uvtri = np.genfromtxt(trifile, dtype=np.int32)
    bar = []
    for i in range(3):
        bar.append(np.genfromtxt(barfiles[i], dtype=np.float32))

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


def test_sizes():
    """Smoke test confirming expected sizes for the encoder"""

    # Load topology
    dotobj = load_obj("assets/face_topology.obj")
    vt, vi, vti = dotobj["vt"], dotobj["vi"], dotobj["vti"]

    import time

    st = time.time()
    uv_shape = 1024
    index_img, bary_img, _ = make_closest_uv_barys(
        torch.from_numpy(vt),
        torch.from_numpy(vti),
        uv_shape,
    )
    print(f"took {time.time()-st:.2f} seconds")
    print(index_img.shape, bary_img.shape)

    expression_encoder = ExpressionEncoder(index_img.numpy(), bary_img.numpy())
    print(expression_encoder)
