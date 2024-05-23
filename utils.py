import json
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple, Union

import einops
import numpy as np
import pandas as pd
import torch as th
from PIL import Image

from data.utils import MugsyCapture
from igl import point_mesh_squared_distance
# rtree and KDTree required by trimesh, though not explicitly in its deps for leanness
# from rtree import Rtree  # noqa
from trimesh import Trimesh
from trimesh.triangles import points_to_barycentric


def closest_point(mesh, points):
    """Helper function that mimics trimesh.proximity.closest_point but uses IGL for faster queries."""
    v = mesh.vertices
    vi = mesh.faces
    dist, face_idxs, p = point_mesh_squared_distance(points, v, vi)
    return p, dist, face_idxs


ObjectType = Dict[str, Union[List[np.ndarray], np.ndarray]]


def tocuda(d: Union[th.Tensor, np.ndarray, Dict, List]) -> Union[th.Tensor, Dict, List]:
    if isinstance(d, th.Tensor):
        return d.to("cuda")
    elif isinstance(d, dict):
        return {k: tocuda(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [tocuda(v) for v in d]
    else:
        return d


def get_renderoptions():
    return dict(dt=1.0)


def get_autoencoder(dataset, assetpath: str):

    import models.autoencoder as aemodel
    import models.bg.mlp2d as bglib
    import models.bottlenecks.vae as vae
    import models.colorcals.colorcal as colorcalib
    import models.decoders.assembler as decoderlib
    import models.encoders.expression as expression_encoder_lib
    import models.encoders.identity as identity_encoder_lib
    import models.raymarchers.mvpraymarcher as raymarcherlib

    allcameras = dataset.get_allcameras()
    ncams = len(allcameras)

    print("@@@ Get autoencoder ABLATION CONFIG FILE : length of data set : {}".format(len(dataset.identities)))
    print(f"dataset vertmean: {dataset.vertmean.shape}")

    vertmean = th.from_numpy(dataset.vertmean)
    vertstd = dataset.vertstd

    # load per-textel triangulation indices
    objpath = f"{assetpath}/face_topology.obj"
    resolution = 1024
    uvdata = create_uv_baridx(objpath, resolution)
    vt, vi, vti = uvdata["uv_coord"], uvdata["tri"], uvdata["uv_tri"]

    # Encoders
    expression_encoder = expression_encoder_lib.ExpressionEncoder(uvdata["uv_idx"], uvdata["uv_bary"])
    id_encoder = identity_encoder_lib.IdentityEncoder(uvdata["uv_idx"], uvdata["uv_bary"], wsize=128)

    # VAE bottleneck for the expression encoder
    bottleneck = vae.VAE_bottleneck(64, 16)

    # Decoder
    volradius = 256.0
    decoder = decoderlib.DecoderAssembler(
        vt=np.array(vt, dtype=np.float32),
        vi=np.array(vi, dtype=np.int32),
        vti=np.array(vti, dtype=np.int32),
        idxim=uvdata["uv_idx"],
        barim=uvdata["uv_bary"],
        vertmean=vertmean,
        vertstd=vertstd,
        volradius=volradius,
        nprims=128 * 128,
        primsize=(8, 8, 8),
    )

    # NOTE(julieta) this ray marcher expects the channels of the template to be last by default
    raymarcher = raymarcherlib.Raymarcher(volradius)
    colorcal = colorcalib.Colorcal(len(dataset.get_allcameras()), len(dataset.identities))
    bgmodel = bglib.BackgroundModelSimple(ncams, len(dataset.identities))

    ae = aemodel.Autoencoder(
        identity_encoder=id_encoder,
        expression_encoder=expression_encoder,
        bottleneck=bottleneck,
        decoder_assembler=decoder,
        raymarcher=raymarcher,
        colorcal=colorcal,
        bgmodel=bgmodel,
    )

    print("id_encoder params:", sum(p.numel() for p in ae.id_encoder.parameters() if p.requires_grad))
    print(f"encoder params: {sum(p.numel() for p in ae.expr_encoder.parameters() if p.requires_grad):_}")
    print(f"decoder params: {sum(p.numel() for p in ae.decoder_assembler.parameters() if p.requires_grad):_}")
    print(f"colorcal params: {sum(p.numel() for p in ae.colorcal.parameters() if p.requires_grad):_}")
    print(f"bgmodel params: {sum(p.numel() for p in ae.bgmodel.parameters() if p.requires_grad):_}")
    print(f"total params: {sum(p.numel() for p in ae.parameters() if p.requires_grad):_}")

    return ae


def load_checkpoint(ae, filename):
    ae = th.nn.DataParallel(ae)
    checkpoint = th.load(filename)
    ae.load_state_dict(checkpoint, strict=True)
    return ae


def load_camera_calibration(path: Union[str, Path]) -> Dict[str, Dict[str, np.ndarray]]:
    """Load a KRT file containing camera parameters
    Args:
        path: File path that contains the KRT information
    Returns:
        A dictionary with
            'intrin'
            'dist'
            'extrin'
    """

    with open(path, "r") as f:
        camera_list = json.load(f)["KRT"]

    cameras = {}

    for item in camera_list:
        camera_name = item["cameraId"]

        RT = np.array(item["T"])
        RT = RT[:4, :3]
        RT = RT.T
        out = {
            "intrin": np.array(item["K"]).T,
            "extrin": RT,
            "dist": np.array(item["distortion"] + [0.0]),
            "model": "radial-tangential",
            "height": 4096,
            "width": 2668,
        }

        cameras[camera_name] = out

    return cameras


def load_obj(path: Union[str, TextIO], return_vn: bool = False) -> ObjectType:
    """Load wavefront OBJ from file. See https://en.wikipedia.org/wiki/Wavefront_.obj_file for file format details
    Args:
        path: Where to load the obj file from
        return_vn: Whether we should return vertex normals

    Returns:
        Dictionary with the following entries
            v: n-by-3 float32 numpy array of vertices in x,y,z format
            vt: n-by-2 float32 numpy array of texture coordinates in uv format
            vi: n-by-3 int32 numpy array of vertex indices into `v`, each defining a face.
            vti: n-by-3 int32 numpy array of vertex texture indices into `vt`, each defining a face
            vn: (if requested) n-by-3 numpy array of normals
    """

    if isinstance(path, str):
        with open(path, "r") as f:
            lines: List[str] = f.readlines()
    else:
        lines: List[str] = path.readlines()

    v = []
    vt = []
    vindices = []
    vtindices = []
    vn = []

    for line in lines:
        if line == "":
            break

        if line[:2] == "v ":
            v.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "vt":
            vt.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "vn":
            vn.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "f ":
            vindices.append([int(entry.split("/")[0]) - 1 for entry in line.split()[1:]])
            if line.find("/") != -1:
                vtindices.append([int(entry.split("/")[1]) - 1 for entry in line.split()[1:]])

    if len(vt) == 0:
        assert len(vtindices) == 0, "Tried to load an OBJ with texcoord indices but no texcoords!"
        vt = [[0.5, 0.5]]
        vtindices = [[0, 0, 0]] * len(vindices)

    # If we have mixed face types (tris/quads/etc...), we can't create a
    # non-ragged array for vi / vti.
    mixed_faces = False
    for vi in vindices:
        if len(vi) != len(vindices[0]):
            mixed_faces = True
            break

    if mixed_faces:
        vi = [np.array(vi, dtype=np.int32) for vi in vindices]
        vti = [np.array(vti, dtype=np.int32) for vti in vtindices]
    else:
        vi = np.array(vindices, dtype=np.int32)
        vti = np.array(vtindices, dtype=np.int32)

    out = {
        "v": np.array(v, dtype=np.float32),
        "vn": np.array(vn, dtype=np.float32),
        "vt": np.array(vt, dtype=np.float32),
        "vi": vi,
        "vti": vti,
    }

    if return_vn:
        assert len(out["vn"]) > 0
        return out
    else:
        out.pop("vn")
        return out


def closest_point_barycentrics(v, vi, points):
    """Given a 3D mesh and a set of query points, return closest point barycentrics
    Args:
        v: np.array (float)
        [N, 3] mesh vertices
        vi: np.array (int)
        [N, 3] mesh triangle indices
        points: np.array (float)
        [M, 3] query points
    Returns:
        Tuple[approx, barys, interp_idxs, face_idxs]
            approx:       [M, 3] approximated (closest) points on the mesh
            barys:        [M, 3] barycentric weights that produce "approx"
            interp_idxs:  [M, 3] vertex indices for barycentric interpolation
            face_idxs:    [M] face indices for barycentric interpolation. interp_idxs = vi[face_idxs]
    """
    mesh = Trimesh(vertices=v, faces=vi)
    p, _, face_idxs = closest_point(mesh, points)

    barys = points_to_barycentric(mesh.triangles[face_idxs], p)
    b0, b1, b2 = np.split(barys, 3, axis=1)

    interp_idxs = vi[face_idxs]
    v0 = v[interp_idxs[:, 0]]
    v1 = v[interp_idxs[:, 1]]
    v2 = v[interp_idxs[:, 2]]
    approx = b0 * v0 + b1 * v1 + b2 * v2
    return approx, barys, interp_idxs, face_idxs


def make_closest_uv_barys(
    vt: th.Tensor,
    vti: th.Tensor,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
    return_approx_dist: bool = False,
) -> Tuple[th.LongTensor, th.FloatTensor, Optional[th.FloatTensor]]:
    """Compute a UV-space barycentric map where each texel contains barycentric
    coordinates for the closest point on a UV triangle.
    Args:
        vt: Texture coordinates. Shape = [n_texcoords, 2]
        vti: Face texture coordinate indices. Shape = [n_faces, 3]
        uv_shape: Shape of the texture map. (HxW)
        flip_uv: Whether to flip UV coordinates along the V axis (OpenGL -> numpy/pytorch convention)
        return_approx_dist: Whether to include the distance to the nearest point
    Returns:
        th.Tensor: index_img: Face index image, shape [uv_shape[0], uv_shape[1]]
        th.Tensor: Barycentric coordinate map, shape [uv_shape[0], uv_shape[1], 3]
        Optional[th.Tensor]: Distances to the nearest point, shape [uv_shape[0], uv_shape[1]]
    """

    if isinstance(uv_shape, int):
        uv_shape = (uv_shape, uv_shape)

    if flip_uv:
        # Flip here because texture coordinates in some of our topo files are
        # stored in OpenGL convention with Y=0 on the bottom of the texture
        # unlike numpy/torch arrays/tensors.
        vt = vt.clone()
        vt[:, 1] = 1 - vt[:, 1]

    # Texel to UV mapping (as per OpenGL linear filtering)
    # https://www.khronos.org/registry/OpenGL/specs/gl/glspec46.core.pdf
    # Sect. 8.14, page 261
    # uv=(0.5,0.5)/w is at the center of texel [0,0]
    # uv=(w-0.5, w-0.5)/w is the center of texel [w-1,w-1]
    # texel = floor(u*w - 0.5)
    # u = (texel+0.5)/w
    uv_grid = th.meshgrid(
        th.linspace(0.5, uv_shape[0] - 1 + 0.5, uv_shape[0]) / uv_shape[0],
        th.linspace(0.5, uv_shape[1] - 1 + 0.5, uv_shape[1]) / uv_shape[1],
        indexing="ij",
    )  # HxW, v,u
    uv_grid = th.stack(uv_grid[::-1], dim=2)  # HxW, u, v

    uv = uv_grid.reshape(-1, 2).data.to("cpu").numpy()
    # pyre-fixme[6]: For 1st param expected `Collection[ndarray]` but got
    #  `Tuple[Tensor, Tensor]`.
    vth = np.hstack((vt, vt[:, 0:1] * 0 + 1))
    uvh = np.hstack((uv, uv[:, 0:1] * 0 + 1))
    approx, barys, interp_idxs, face_idxs = closest_point_barycentrics(vth, vti, uvh)
    index_img = th.from_numpy(face_idxs.reshape(uv_shape[0], uv_shape[1])).long()
    bary_img = th.from_numpy(barys.reshape(uv_shape[0], uv_shape[1], 3)).float()

    if return_approx_dist:
        dist = np.linalg.norm(approx - uvh, axis=1)
        dist = th.from_numpy(dist.reshape(uv_shape[0], uv_shape[1])).float()
        # pyre-fixme[7]: Expected `Tuple[LongTensor, FloatTensor,
        # Optional[FloatTensor]]]` but got `Tuple[Tensor, Tensor, Tensor]`.
        return index_img, bary_img, dist
    else:
        # pyre-fixme[7]: Expected `Tuple[LongTensor, FloatTensor,
        # Optional[FloatTensor]]]` but got `Tuple[Tensor, Tensor, NoneType]`.
        return index_img, bary_img, None


def create_uv_baridx(objpath: str, resolution: int = 1024):
    """
    TODO(julieta) document
    """

    dotobj = load_obj(objpath)
    vt, vi, vti = dotobj["vt"], dotobj["vi"], dotobj["vti"]

    index_img, bary_img, _ = make_closest_uv_barys(
        th.from_numpy(vt),
        th.from_numpy(vti),
        uv_shape=resolution,
        flip_uv=False,
    )
    bary_img = einops.rearrange(bary_img, "H W C -> C H W")

    index_img = index_img.numpy()
    bary_img = bary_img.numpy()

    idx0 = np.flipud(vi[index_img, 0])
    idx1 = np.flipud(vi[index_img, 1])
    idx2 = np.flipud(vi[index_img, 2])
    bar0 = np.flipud(bary_img[0])
    bar1 = np.flipud(bary_img[1])
    bar2 = np.flipud(bary_img[2])

    return {
        "uv_idx": np.concatenate((idx0[None, :, :], idx1[None, :, :], idx2[None, :, :]), axis=0),
        "uv_bary": np.concatenate((bar0[None, :, :], bar1[None, :, :], bar2[None, :, :]), axis=0),
        "uv_coord": vt,
        "uv_tri": vti,
        "tri": vi,
    }


def render_img(listsofimages, outpath) -> None:
    """saves image given a list of list of images

    Args:
        listsofimages (List[List[np.array]]): list of list of images
        outpath (str): path to save the image
    Returns:
        Nothing, the image is saved to outpath
    """

    combined_imgs = []
    for images in listsofimages:
        rgb = np.hstack(images)
        combined_imgs.append(rgb)

    rgb = np.vstack(combined_imgs)

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb_img = Image.fromarray(rgb)
    rgb_img.save(outpath)


def train_csv_loader(base_dir: Path, csv_path: Path, nids: int) -> Tuple[List[MugsyCapture], List[Path]]:
    """loads train data by id given the csv file of ids
    Args:
        base_dir (Path): base directory for the dataset
        csv_path (Path): id csv file path
        nids (int): number of ids to select from

    Returns:
        List[MugsyCapture]: train_captures: list of mugsy captures
        List[Path]: train_dirs: list of directories for all captures
    """
    df = pd.read_csv(csv_path, dtype=str)[:nids]

    train_captures = []
    train_dirs = []

    for capture in df.itertuples():
        capture = MugsyCapture(mcd=capture.mcd, mct=capture.mct, sid=capture.sid)
        train_captures.append(capture)

        capture_dir = f"{base_dir}/{capture.folder_name()}/decoder"
        train_dirs.append(capture_dir)

    return train_captures, train_dirs
