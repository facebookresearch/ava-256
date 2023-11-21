import torch
import torch.nn as nn
import torch.nn.functional as F

class Raymarcher(nn.Module):
    def __init__(self, volsampler):
        super(Raymarcher, self).__init__()

        self.volsampler = volsampler

    # TODO: pass viewpos/raydir
    # TODO: what is viewpos/viewrot for? calculating depth of ray
    def forward(self, raypos, raydir, viewpos, viewrot, tminmax, decout,
            encoding=None, rastdepth=None, renderoptions={}, **kwargs):

        # rescale world
        viewpos = viewpos / self.volradius
        if rastdepth is not None:
            rastdepth = rastdepth / self.volradius
        renderoptions["dt"] = renderoptions["dt"] / self.volradius

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
        tmin = tmin - renderoptions["dt"] * torch.rand_like(tmin)

        raypos = viewpos[:, None, None, :] + raydir * 0.#t[..., None] # NHWC
        rayposend = viewpos[:, None, None, :] + raydir * tmax[:, :, :, None]
        tminmax = torch.stack([tmin, tmax], dim=-1)

        # TODO: world scale

        # TODO: raymarcherinput instead of decout?

        # TODO: option for stopping at mesh, etc

        # TODO: use tminmax

        t = tminmax[..., 0] + 0.
        raypos = raypos + raydir * t[..., None]

        rayrgb = torch.zeros_like(raypos.permute(0, 3, 1, 2)) # NCHW
        if "multaccum" in renderoptions and renderoptions["multaccum"]:
            lograyalpha = torch.zeros_like(rayrgb[:, 0:1, :, :]) # NCHW
        else:
            rayalpha = torch.zeros_like(rayrgb[:, 0:1, :, :]) # NCHW

        # TMP
        #nsteps = torch.zeros_like(rayalpha)

        # raymarch
        done = torch.zeros_like(t).bool()
        #i = 0
        while not done.all():
            if rastdepth is not None:
                raydepth = torch.sum((raypos - viewpos[:, None, None, :]) * viewrot[:, None, None, 2, :], dim=-1)

            valid = torch.prod(torch.gt(raypos, -1.0) * torch.lt(raypos, 1.0), dim=-1).byte()
            validf = valid.float()

            if "terminateatmesh" in renderoptions and renderoptions["terminateatmesh"]:
                validf = validf * (raydepth < rastdepth).float()

            validf = validf[:, None, :, :]

            sample_rgb, sample_alpha = self.volsampler(
                    raypos[:, None, :, :, :], raydir[:, None, :, :, :],
                    **decout,
                    encoding=encoding,
                    renderoptions=renderoptions)

            sample_rgb = sample_rgb[:, :, 0, :, :]
            sample_alpha = sample_alpha[:, :, 0, :, :]

            jitter = torch.exp(renderoptions["stepjitter"] * torch.randn_like(t))
            step = renderoptions["dt"] * jitter

            if "terminateatmesh" in renderoptions and renderoptions["terminateatmesh"]:
                done = done | ((t + step) >= tminmax[..., 1]) | (raydepth >= rastdepth)
            else:
                done = done | ((t + step) >= tminmax[..., 1])

            if "multaccum" in renderoptions and renderoptions["multaccum"]:
                contrib = torch.exp(-lograyalpha) * (1. - torch.exp(-sample_alpha * step[:, None, :, :] * validf))

                rayrgb = rayrgb + sample_rgb * contrib
                lograyalpha = lograyalpha + sample_alpha * step[:, None, :, :] * validf
            else:
                contrib = ((rayalpha + sample_alpha * step[:, None, :, :]).clamp(max=1.) - rayalpha) * validf

                rayrgb = rayrgb + sample_rgb * contrib
                rayalpha = rayalpha + contrib

            #print(i, raypos[0, 512, 334, :].data.to("cpu").numpy())
            raypos = raypos + raydir * step[:, :, :, None]
            t = t + step
            #i += 1

            # TMP
            #nsteps += validf

        #print("!", nsteps.min().item(), nsteps.max().item(), nsteps.mean().item(), nsteps[0, 0, 512, 334].item())

        if "multaccum" in renderoptions and renderoptions["multaccum"]:
            rayalpha = 1. - torch.exp(-lograyalpha)

        return rayrgb, rayalpha

