
import torch

def generate_geomap(geo: torch.Tensor, uv_tidx: torch.Tensor, uv_bary: torch.Tensor) -> torch.Tensor:
    """
    Create a geometry image given a series of vertices and their topology

    Args:
        geo: n-by-3 tensor with vertices
        uv_tidx:
        uv_bary:
    Returns:
      An n-by-2 tensor representing

    """
    n = geo.shape[0]
    g = geo.view(n, -1, 3).permute(0, 2, 1)
    geomap = (g[:,:,uv_tidx[0]] * uv_bary[0][None,None,:,:] + \
              g[:,:,uv_tidx[1]] * uv_bary[1][None,None,:,:] + \
              g[:,:,uv_tidx[2]] * uv_bary[2][None,None,:,:])
    return geomap
