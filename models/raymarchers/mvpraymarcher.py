"""
Raymarcher for a mixture of volumetric primitives
"""
import os
import itertools
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from extensions.mvpraymarch.mvpraymarch import mvpraymarch as mvpraymarch

class Raymarcher(nn.Module):
    def __init__(self, volradius, autotuneperf=False):
        super(Raymarcher, self).__init__()

        self.volradius = volradius
        self.autotuneperf = autotuneperf
        self.perfconfigs = [dict(x) for x in itertools.product(
            [("usebvh", False), ("usebvh", True)],
            [("algo", 0), ("algo", 1)],
            [("blocksize", (8, 8)), ("blocksize", (12, 12)), ("blocksize", 256), ("blocksize", 512)])]

    #def forward(self, raypos, raydir, viewpos, viewrot, tminmax, decout,
    #        encoding=None, rastdepth=None, renderoptions={}, **kwargs):
    def forward(self, viewpos, viewrot, raydir, decout,
            encoding=None, rastdepth=None, renderoptions={}, trainiter=-1,
            **kwargs):

        for k in ["terminateatmesh", "stepjitter", "multaccum"]:
            if k in renderoptions:
                print("Raymarcher doesn't support {}", k)

        # rescale world
        viewpos = viewpos / self.volradius
        if rastdepth is not None:
            rastdepth = rastdepth / self.volradius
        renderoptions["dt"] = renderoptions["dt"] / self.volradius
        #decout["primpos"] = decout["primpos"] / self.volradius
        #decout["primscale"] = decout["primscale"] / self.volradius

        # compute raymarching starting points
        with torch.no_grad():
            # TODO: volbounds
            t1 = (-1. - viewpos[:, None, None, :]) / raydir
            t2 = ( 1. - viewpos[:, None, None, :]) / raydir
            tmin = torch.max(torch.min(t1[..., 0], t2[..., 0]),
                   torch.max(torch.min(t1[..., 1], t2[..., 1]),
                             torch.min(t1[..., 2], t2[..., 2])))
            tmax = torch.min(torch.max(t1[..., 0], t2[..., 0]),
                   torch.min(torch.max(t1[..., 1], t2[..., 1]),
                             torch.max(t1[..., 2], t2[..., 2])))

            intersections = tmin < tmax
            t = torch.where(intersections, tmin, torch.zeros_like(tmin)).clamp(min=0.)
            tmin = torch.where(intersections, tmin, torch.zeros_like(tmin))
            tmax = torch.where(intersections, tmax, torch.zeros_like(tmin))

        # random starting point
        #t = t - self.dt * torch.rand_like(t)
        #tmin = tmin - renderoptions["dt"] * torch.rand_like(tmin)

        raypos = viewpos[:, None, None, :] + raydir * 0.#t[..., None] # NHWC
        rayposbeg = viewpos[:, None, None, :] + raydir * tmin[:, :, :, None]
        rayposend = viewpos[:, None, None, :] + raydir * tmax[:, :, :, None]
        tminmax = torch.stack([tmin, tmax], dim=-1)

        #if not os.path.exists("/mnt/captures/stephenlombardi/tmpdata.npz"):
        #    np.savez("/mnt/captures/stephenlombardi/tmpdata.npz", **{
        #        "raypos": raypos.data.to("cpu").numpy(),
        #        "raydir": raydir.data.to("cpu").numpy(),
        #        "tminmax": tminmax.data.to("cpu").numpy(),
        #        "template": decout["template"].data.to("cpu").numpy(),
        #        "primpos": decout["primpos"].data.to("cpu").numpy(),
        #        "primrot": decout["primrot"].data.to("cpu").numpy(),
        #        "primscale": decout["primscale"].data.to("cpu").numpy()})

        # automatically tune params periodically
        if self.autotuneperf and trainiter % 1000 == 0:
            besttime = 1000000.
            bestconfig = None
            for config in self.perfconfigs:
                nruns = 100
                totaltime = 0.
                with torch.no_grad():
                    for i in range(nruns):
                        torch.cuda.synchronize()
                        t0 = time.time()
                        rayrgba = mvpraymarch(raypos, raydir, renderoptions["dt"], tminmax,
                                template=decout["template"],
                                warp=decout["warp"] if "warp" in decout else None,
                                primpos=decout["primpos"],
                                primrot=decout["primrot"],
                                primscale=decout["primscale"],
                                **config)
                        torch.cuda.synchronize()
                        t1 = time.time()
                        totaltime += t1 - t0
            if totaltime < besttime:
                besttime = totaltime
                bestconfig = config
            print("best config:", bestconfig, totaltime / nruns)

        rayrgba = mvpraymarch(raypos, raydir, renderoptions["dt"], tminmax,
                template=decout["template"],
                warp=decout["warp"] if "warp" in decout else None,
                primpos=decout["primpos"],
                primrot=decout["primrot"],
                primscale=decout["primscale"])
        
        return rayrgba.permute(0, 3, 1, 2), rayposbeg, rayposend
