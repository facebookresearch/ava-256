"""
Raymarcher for a mixture of volumetric primitives
"""
from typing import Dict, Optional, Tuple

import torch as th

from extensions.mvpraymarch.mvpraymarch import mvpraymarch


class Raymarcher(th.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dt = float(config.render.raymarcher_options.get("dt", 1.0))
        self.fadescale = float(config.render.raymarcher_options.get("fadescale", 8.0))
        self.fadeexp = float(config.render.raymarcher_options.get("fadeexp", 8.0))
        self.usebvh = config.render.raymarcher_options.get("usebvh", "fixedorder")
        self.accum = int(config.render.raymarcher_options.get("accum", 0))
        self.termthresh = float(config.render.raymarcher_options.get("termthresh", 0.0))
        self.chlast = config.render.raymarcher_options.get("chlast", True)
        self.volume_radius = float(config.render.raymarcher_options.get("volume_radius", 256))

        self.primintscale = float(config.render.get("primintscale", 1.0))
        self.use_primint = bool(config.train.loss_weights.get("primint", False))
        self.use_integprior = bool(config.train.loss_weights.get("integprior", False))
        self.fixed_bvh_cache = {-1: (th.empty(0), th.empty(0), th.empty(0))}

    def forward(
        self, raypos, raydir, tminmax, outputs: Dict[str, th.Tensor], with_pos_img: bool = False
    ) -> Tuple[Optional[th.Tensor], Optional[th.Tensor], th.Tensor, Optional[th.Tensor]]:
        # Filter out any prims that were generated but not used (ex: if we
        # decode prims from a texture atlas w/empty space, the prims whose
        # centers come from empty space will be invalid).
        outputs = dict(outputs)
        if "valid_prims" in outputs:
            valid_prims = outputs["valid_prims"]
            assert valid_prims.shape[0] == outputs["template"].shape[1]

            outputs["template"] = outputs["template"][:, valid_prims].contiguous()
            outputs["primpos"] = outputs["primpos"][:, valid_prims].contiguous()
            outputs["primrot"] = outputs["primrot"][:, valid_prims].contiguous()
            outputs["primscale"] = outputs["primscale"][:, valid_prims].contiguous()
            if "warp" in outputs:
                outputs["warp"] = outputs["warp"][:, valid_prims].contiguous()

        dt = self.dt / self.volume_radius

        warp: Optional[th.Tensor] = None
        if "warp" in outputs:
            warp = outputs["warp"]

        accum: int = 0 if th.jit.is_scripting() else self.accum
        blocksize: Tuple[int, int] = (8, 16)

        if accum == 0 and self.accum == 2:
            # We can use a model trained w/accum == 2 as if it were additive
            # (accum == 0) so long as the termthresh makes it close to
            # additive.
            assert self.termthresh > 0.9

        rayrgba, t_img = mvpraymarch(
            raypos,
            raydir,
            dt,
            tminmax,
            template=outputs["template"],
            warp=warp,
            primpos=outputs["primpos"],
            primrot=outputs["primrot"],
            primscale=outputs["primscale"],
            fadescale=self.fadescale,
            fadeexp=self.fadeexp,
            usebvh=self.usebvh,
            accum=accum,
            termthresh=self.termthresh,
            chlast=self.chlast,
            with_t_img=with_pos_img,
            fixed_bvh_cache=self.fixed_bvh_cache,
            blocksize=blocksize,
        )

        assert not isinstance(rayrgba, tuple)

        # OpenGL expects channels-last, so we can save copying if we don't
        # permute here. The realtime system should only ever use the full rgba
        # rendered output in channels-last format w/no splitting or anything.
        if th.jit.is_scripting():
            rayrgb, rayalpha = None, None
        else:
            rayrgba = rayrgba.permute(0, 3, 1, 2)
            rayrgb, rayalpha = rayrgba[:, :3].contiguous(), rayrgba[:, 3:4].contiguous()

        pos_img: Optional[th.Tensor] = None
        if with_pos_img:
            assert t_img is not None
            pos_img = (raypos + raydir * t_img[..., None]) * self.volume_radius

        return rayrgb, rayalpha, rayrgba, pos_img
