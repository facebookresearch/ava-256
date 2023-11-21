import torch
import torch.nn as nn
import torch.nn.functional as F

class VolSampler(nn.Module):
    def __init__(self, displacementwarp=False):
        super(VolSampler, self).__init__()

        self.displacementwarp = displacementwarp

    def forward(self, raypos, raydir, template, warp=None, gwarps=None, gwarprot=None, gwarpt=None, renderoptions={}, **kwargs):
        valid = None
        if "viewtemplate" not in renderoptions or not renderoptions["viewtemplate"]:
            if gwarps is not None:
                raypos = (torch.sum(
                    (raypos - gwarpt[:, None, None, None, :])[:, :, :, :, None, :] *
                    gwarprot[:, None, None, None, :, :], dim=-1) *
                    gwarps[:, None, None, None, :])
            if warp is not None:
                if self.displacementwarp:
                    raypos = raypos + F.grid_sample(warp, raypos, align_corners=True).permute(0, 2, 3, 4, 1)
                else:
                    valid = torch.prod((raypos > -1.) * (raypos < 1.), dim=-1).float()
                    raypos = F.grid_sample(warp, raypos, align_corners=True).permute(0, 2, 3, 4, 1)
        val = F.grid_sample(template, raypos, align_corners=True)
        if valid is not None:
            val = val * valid[:, None, :, :, :]
        return val[:, :3, :, :, :], val[:, 3:, :, :, :]
