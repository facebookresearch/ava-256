import torch


def generate_geomap(geo: torch.Tensor, uv_tidx: torch.Tensor, uv_bary: torch.Tensor) -> torch.Tensor:
    """
    Create a geometry image given a series of vertices and their topology
    Args:
        geo: [N, 3] tensor with vertices
        uv_tidx: [M, M] tensor with geometry image indices
        uv_bary: [M, M, 3] tensor with geometry image barycentric coordinates
    Returns:
      An [M, M, 2] tensor representing the input geometry as an image

    NOTE(julieta) if the source topology of `uv_tidx` and `uv_bary` is low res but the images themselves are high res,
      `uv_tidx` will contain multiple repeated indices, which means that backpropagating through this function would be
      very very slow (see https://github.com/pytorch/pytorch/issues/41162#issuecomment-655834491).
    """

    n = geo.shape[0]
    g = geo.view(n, -1, 3).permute(0, 2, 1)
    geomap = (
        g[:, :, uv_tidx[0]] * uv_bary[0][None, None, :, :]
        + g[:, :, uv_tidx[1]] * uv_bary[1][None, None, :, :]
        + g[:, :, uv_tidx[2]] * uv_bary[2][None, None, :, :]
    )
    return geomap
