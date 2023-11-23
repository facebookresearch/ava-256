import torch.nn as nn


###############################################################################
# add per-pixel gains as well as bias
class DecoderSlab(nn.Module):
    def __init__(
        self,
        imsize: int,
        nboxes: int,
        boxsize: int,
        outch: int,
        viewcond: bool = False,
        texwarp: bool = False,
    ):
        super(DecoderSlab, self).__init__()

        self.imsize = imsize
        self.nboxes = nboxes
        self.boxsize = boxsize
        self.outch = outch
        self.texwarp = texwarp
        self.viewcond = viewcond

        nh = int(np.sqrt(self.nboxes))
        assert nh * nh == self.nboxes
        if nh == 512:
            assert boxsize == 2
        elif nh == 64:
            assert boxsize == 16
        elif nh == 128:
            assert boxsize == 8
        else:
            print(f"boxsize {boxsize} not supported yet")

        l = models.utils.LinearWN
        # c = models.utils.ConvTranspose2dWN
        c = models.utils.ConvTranspose2dWNUB
        v = models.utils.Conv2dWN
        # a = CenteredLeakyReLU
        a = nn.LeakyReLU
        s = nn.Sequential

        self.encmod = s(v(16, 16, 1, 1, 0), a(0.2, inplace=True))
        inch = 16 + 16
        if self.viewcond:
            self.viewmod = s(l(3, 16), a(0.2, inplace=True), l(16, 8 * 4 * 4), a(0.2, inplace=True))
            inch += 8

        if imsize == 1024:
            size = [inch, 256, 128, 128, 64, 64, 32, 16, self.boxsize * self.outch]
            scale_factor = 4
        elif imsize == 512:
            size = [inch, 256, 128, 128, 64, 64, 32, self.boxsize * self.outch]
            scale_factor = 2
        else:
            print(f"Unsupported image size: {size}")
            quit()
        self.nlayers = len(size) - 1

        h = 8
        for i in range(self.nlayers):
            # t = [c(size[i], size[i+1], 4, 2, 1)]
            t = [c(size[i], size[i + 1], h, h, 4, 2, 1)]
            h *= 2

            if i < self.nlayers - 1:
                t.append(a(0.2, inplace=True))
            self.add_module(f"t{i}", s(*t))

        if self.texwarp:
            self.warpmod = s(
                v(inch, 256, 1, 1, 0),
                a(0.2, inplace=True),
                c(256, 256, 4, 2, 1),
                a(0.2, inplace=True),
                c(256, 128, 4, 2, 1),
                a(0.2, inplace=True),
                c(128, 128, 4, 2, 1),
                a(0.2, inplace=True),
                c(128, 64, 4, 2, 1),
                a(0.2, inplace=True),
                c(64, 64, 4, 2, 1),
                a(0.2, inplace=True),
                c(64, 2, 4, 2, 1),
                nn.Upsample(scale_factor=scale_factor, mode="bilinear"),
            )

        # self.apply(lambda x: he_init(x, 0.2))
        # he_init(self._modules[f't{self.nlayers-1}'][-1], 1)
        # if self.texwarp:
        #    he_init(self.warpmod[-2], 1)

        if self.viewcond:
            models.utils.initseq(self.viewmod)
        models.utils.initseq(self.encmod)
        for i in range(self.nlayers):
            models.utils.initseq(self._modules[f"t{i}"])
        if self.texwarp:
            models.utils.initseq(self.warpmod)

        self.bias = nn.Parameter(torch.zeros(self.boxsize * self.outch, imsize, imsize))
        self.bias.data.zero_()

        xgrid, ygrid = np.meshgrid(np.linspace(-1.0, 1.0, imsize), np.linspace(-1.0, 1.0, imsize))
        grid = np.concatenate((xgrid[None, :, :], ygrid[None, :, :]), axis=0)[None, ...].astype(np.float32)
        self.register_buffer("warpbias", torch.from_numpy(grid))

    def forward(self, ex_enc, id_enc, id_gainbias, view=None, use_warp=True, iternum=-1):
        z = self.encmod(ex_enc).view(-1, 16, 4, 4)
        x = torch.cat([z, id_enc], dim=1) if id_enc is not None else z

        if self.viewcond:
            v = self.viewmod(view).view(-1, 8, 4, 4)
            x = torch.cat([v, x], dim=1)
        x_orig = x

        ###############################################################################################################################
        scale = 1 / sqrt(2)
        for i in range(self.nlayers):
            xx = self._modules[f"t{i}"](x)

            if id_gainbias is not None:
                n = id_gainbias[i].shape[1] // 2
                if n == xx.shape[1]:
                    x = (xx * (id_gainbias[i][:, :n, ...] * 0.01 + 1.0) + id_gainbias[i][:, n:, ...]) * scale
                elif n * 2 == xx.shape[1]:
                    x = (xx + id_gainbias[i]) * scale
                else:
                    x = xx  # note: last layer (1024x1024) ignores the pass through since slab is larger than 3 channels
            else:
                x = xx

        # #test without skip connections
        # for i in range(self.nlayers):
        #     x = self._modules[f't{i}'](x)
        ###############################################################################################################################

        if self.texwarp and use_warp:
            w = self.warpmod(x_orig)
            w = w / self.imsize + self.warpbias
            x = torch.nn.functional.grid_sample(x, w.permute(0, 2, 3, 1))
        else:
            w = None
        tex = x + self.bias[None, :, :, :]

        x0 = tex
        x = tex
        h = int(np.sqrt(self.nboxes))
        w = int(h)
        x = x.view(x.size(0), self.boxsize, self.outch, h, self.boxsize, w, self.boxsize)
        x = x.permute(0, 3, 5, 2, 1, 4, 6)

        x = x.reshape(x.size(0), self.nboxes, self.outch, self.boxsize, self.boxsize, self.boxsize)

        return x
