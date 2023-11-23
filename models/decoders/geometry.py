###############################################################################
# add per-pixel gains as well as bias
# add geometry and motion as output, remove warp option, remove viewcond option
# positivity for alpha
class DecoderGeoSlab2(nn.Module):
    def __init__(self, uv, tri, uvtri, nvtx, motion_size, geo_size, imsize, nboxes, boxsize, disable_id_encoder=False):
        super(DecoderGeoSlab2, self).__init__()

        assert motion_size < imsize
        assert geo_size < imsize

        self.motion_size = motion_size
        self.geo_size = geo_size
        self.imsize = imsize
        self.nboxes = nboxes
        self.boxsize = boxsize

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
        c = models.utils.ConvTranspose2dWNUB
        v = models.utils.Conv2dWN
        a = nn.LeakyReLU
        s = nn.Sequential

        # reduce noise effect of latent expression code
        self.encmod = s(v(16, 16, 1, 1, 0), a(0.2, inplace=True))
        models.utils.initseq(self.encmod)

        inch = 16 + 16 if not disable_id_encoder else 16  # first is for expression, second for identity

        if imsize == 1024:
            size = [inch, 256, 128, 128, 64, 64, 32, 16, self.boxsize]
            scale_factor = 4
        elif imsize == 512:
            size = [inch, 256, 128, 128, 64, 64, 32, self.boxsize]
            scale_factor = 2
        else:
            print(f"Unsupported image size: {size}")
            quit()
        self.nlayers = len(size) - 1

        # build deconv arch with early exists for geometry and motion
        h = 8
        for i in range(self.nlayers):
            t = [c(size[i], size[i + 1], h, h, 4, 2, 1)]
            if i < self.nlayers - 1:
                t.append(a(0.2, inplace=True))
            self.add_module(f"t{i}", s(*t))
            models.utils.initseq(self._modules[f"t{i}"])

            if h == motion_size:
                self.motion = s(v(size[i + 1], 64, 1, 1, 0), a(0.2, inplace=True), v(64, 9, 1, 1, 0))
                models.utils.initseq(self.motion)

            if h == geo_size:
                self.geo = s(v(size[i + 1], 64, 1, 1, 0), a(0.2, inplace=True), v(64, 3, 1, 1, 0))
                models.utils.initseq(self.geo)

            h *= 2

        self.bias = nn.Parameter(torch.zeros(self.boxsize, imsize, imsize))
        self.bias.data.zero_()

        xgrid, ygrid = np.meshgrid(np.linspace(-1.0, 1.0, imsize), np.linspace(-1.0, 1.0, imsize))
        grid = np.concatenate((xgrid[None, :, :], ygrid[None, :, :]), axis=0)[None, ...].astype(np.float32)
        self.register_buffer("warpbias", torch.from_numpy(grid))

        # create cropping coordinates for geometry points
        vlists = [list() for _ in range(nvtx)]

        print(f"{nvtx=}")
        print(f"{tri.shape=}")
        print(f"{uvtri.shape=}")

        try:
            for fi in range(tri.shape[0]):
                for fv in range(3):
                    vlists[tri[fi, fv]].append(uvtri[fi, fv])
        except IndexError:
            print(f"{fi=}")
            print(f"{fv=}")
            print(f"{tri[fi,fv]=}")
            print(f"{uvtri[fi,fv]=}")
            raise
        nMaxUVsPerVertex = np.max([len(v) for v in vlists])
        print("Max UVs per vertex: {}".format(nMaxUVsPerVertex))
        nMaxUVsPerVertex = 1  # 2
        uvspervert = np.zeros((nvtx, nMaxUVsPerVertex), dtype=np.int32)
        wuvspervert = np.zeros((nvtx, nMaxUVsPerVertex), dtype=np.float32)
        uvmask = np.ones((nvtx,), dtype=np.float32)
        for tvi in range(len(vlists)):
            if not (len(vlists[tvi])):
                uvmask[tvi] = 0
                continue
            for vsi in range(nMaxUVsPerVertex):
                if vsi < len(vlists[tvi]):
                    uvspervert[tvi, vsi] = vlists[tvi][vsi]
                    wuvspervert[tvi, vsi] = 1.0 / nMaxUVsPerVertex
                elif len(vlists[tvi]):
                    uvspervert[tvi, vsi] = vlists[tvi][0]
                    wuvspervert[tvi, vsi] = 1.0 / nMaxUVsPerVertex
        self.register_buffer("t_nl_uvspervert", torch.from_numpy(uvspervert).long().to("cuda"))
        self.register_buffer("t_nl_wuvspervert", torch.from_numpy(wuvspervert).to("cuda"))
        t_nl_geom_vert_uvs = torch.from_numpy(uv).to("cuda")[self.t_nl_uvspervert, :]
        coords = t_nl_geom_vert_uvs.view(1, -1, nMaxUVsPerVertex, 2) * 2 - 1.0
        self.register_buffer("coords", coords)

    def forward(self, ex_enc, id_enc, id_gainbias, iternum=-1):
        z = self.encmod(ex_enc).view(-1, 16, 4, 4)
        x = torch.cat([z, id_enc], dim=1) if id_enc is not None else z

        scale = 1 / sqrt(2)
        for i in range(self.nlayers):
            xx = self._modules[f"t{i}"](x)

            if id_gainbias is not None:
                n = id_gainbias[i].shape[1] // 2
                if n == xx.shape[1]:
                    x = (xx * (id_gainbias[i][:, :n, ...] * 0.1 + 1.0) + id_gainbias[i][:, n:, ...]) * scale
                elif n * 2 == xx.shape[1]:
                    x = (xx + id_gainbias[i]) * scale
                else:
                    x = xx  # note: last layer (1024x1024) ignores the pass through since slab is larger than 3 channels
            else:
                x = xx

            if x.shape[-1] == self.motion_size:
                mot = self.motion(x)
            if x.shape[-1] == self.geo_size:
                geo = self.geo(x)

        tex = torch.exp((x + self.bias[None, :, :, :]) * 0.1)

        # get motion
        mot = mot.view(mot.size(0), 9, -1).permute(0, 2, 1).contiguous()
        primposresid = mot[:, :, 0:3] * 0.01
        primrvecresid = mot[:, :, 3:6] * 0.01
        primscaleresid = torch.exp(0.01 * mot[:, :, 6:9])

        # get geometry
        coords = self.coords.expand((geo.size(0), -1, -1, -1))
        geo = F.grid_sample(geo, coords).mean(dim=3).permute(0, 2, 1)

        x0 = tex
        x = tex
        h = int(np.sqrt(self.nboxes))
        w = int(h)
        x = x.view(x.size(0), self.boxsize, 1, h, self.boxsize, w, self.boxsize)
        x = x.permute(0, 3, 5, 2, 1, 4, 6)

        x = x.reshape(x.size(0), self.nboxes, 1, self.boxsize, self.boxsize, self.boxsize)

        return x, geo, primposresid, primrvecresid, primscaleresid
