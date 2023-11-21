import torch
import torch.nn as nn

import models.utils

from math import sqrt

import torch.nn.functional as F

####################################################################################################################################
def kl_loss_stable(mu, logstd):
    return torch.mean(-0.5 + torch.abs(logstd) + 0.5 * mu ** 2 + 0.5 * torch.exp(2*-1*torch.abs(logstd)), dim=-1)

####################################################################################################################################
def generate_geomap(geo, uv_tidx, uv_bary):
    n = geo.shape[0]
    g = geo.view(n, -1, 3).permute(0, 2, 1)
    geomap = (g[:,:,uv_tidx[0]] * uv_bary[0][None,None,:,:] + \
              g[:,:,uv_tidx[1]] * uv_bary[1][None,None,:,:] + \
              g[:,:,uv_tidx[2]] * uv_bary[2][None,None,:,:])
    return geomap


####################################################################################################################################
class EncoderExpression(nn.Module):
    def __init__(self, uv_tidx, uv_bary, encoder_channel_mult=1):
        super(EncoderExpression, self).__init__()

        self.register_buffer('uv_tidx', torch.from_numpy(uv_tidx).type(torch.LongTensor))
        self.register_buffer('uv_bary', torch.from_numpy(uv_bary).type(torch.FloatTensor))
        self.C = encoder_channel_mult
        C = self.C

        #l = models.utils.LinearWN
        c = models.utils.Conv2dWN#UB #################################### UB?????????????????????????????????
        #a = lambda: CenteredLeakyReLU(0.2, inplace=True)
        a = lambda: nn.LeakyReLU(0.2, inplace=True)
        s = nn.Sequential

        self.tex = s(c(  3,    16*C, 4, 2, 1), a(),#1024,512
                     c( 16*C,  32*C, 4, 2, 1), a(),#512,256
                     c( 32*C,  64*C, 4, 2, 1), a(),#256,128
                    )

        self.geo = s(c(  3,   16*C, 4, 2, 1), a(),#1024,512
                     c( 16*C, 32*C, 4, 2, 1), a(),#512,256
                     c( 32*C, 32*C, 4, 2, 1), a(),#256,128
                    )

        self.comb = s(c((64+32)*C,  128*C, 4, 2, 1), a(),#128,64
                      c(  128*C,  256*C, 4, 2, 1), a(),#64,32
                      c(  256*C,  256*C, 4, 2, 1), a(),#32,16
                      c(  256*C,  512*C, 4, 2, 1), a(),#16,8
                      c(  512*C,  256*C, 3, 1, 1), a(),#no change in res
                      c(  256*C,  128*C, 3, 1, 1), a(),#no change in res
                      c(  128*C,   64*C, 3, 1, 1), a(),#no change in res
                    # c(  512*C,   64, 4, 2, 1), a(),#8,4
                      c(  64*C,    64, 4, 2, 1), a(),#8,4
                    )

        self.mu = c(64, 16, 1, 1, 0)
        self.logstd = c(64, 16, 1, 1, 0)


        models.utils.initseq(self.tex)
        models.utils.initseq(self.geo)
        models.utils.initseq(self.comb)
        models.utils.initmod(self.mu)
        models.utils.initmod(self.logstd)

    def forward(self, verts, avgtex, neut_verts, neut_avgtex, losslist):

        geo = self.geo(generate_geomap(verts - neut_verts, self.uv_tidx, self.uv_bary))
        tex = self.tex(avgtex - neut_avgtex)
        x = self.comb(torch.cat((tex, geo), dim=1))
        mu, logstd = self.mu(x) * 0.1, self.logstd(x) * 0.01

        if self.training:
            z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)
        else:
            z = mu

        losses = {}
        if "kldiv" in losslist:
            #losses["kldiv"] = torch.mean(-0.5 - logstd + 0.5 * mu ** 2 + 0.5 * torch.exp(2 * logstd), dim=-1)
            losses['kldiv'] = kl_loss_stable(mu, logstd)

        return {"encoding": z}, losses


####################################################################################################################################
#add warp field
import numpy as np
class EncoderIdentity2(nn.Module):
    def __init__(self, uv_tidx, uv_bary, wsize=128):
        super(EncoderIdentity2, self).__init__()

        self.register_buffer('uv_tidx', torch.from_numpy(uv_tidx).type(torch.LongTensor))
        self.register_buffer('uv_bary', torch.from_numpy(uv_bary).type(torch.FloatTensor))

        self.tex = EncoderUNet()
        self.geo = EncoderUNet()
        self.comb = GeoTexCombiner()


        self.wsize = wsize
        xgrid, ygrid = np.meshgrid(np.linspace(-1.0, 1.0, wsize), np.linspace(-1.0, 1.0, wsize))
        grid = np.concatenate((xgrid[None,:, :], ygrid[None,:, :]), axis=0)[None, ...].astype(np.float32)
        self.register_buffer("warpidentity", torch.from_numpy(grid))
        self.bias = nn.Parameter(torch.zeros(1, 2, wsize, wsize))
        self.bias.data.zero_()

    def forward(self, neut_verts, neut_avgtex, losslist = None):

        geo = generate_geomap(neut_verts, self.uv_tidx, self.uv_bary)
        z_geo, b_geo = self.geo(geo)
        z_tex, b_tex = self.tex(neut_avgtex)
        b_geo, b_tex = self.comb(b_geo, b_tex)

        warp = self.warpidentity + self.bias / self.wsize
        for i in range(len(b_geo)):
            w, h, b = b_geo[i].shape[-1], b_geo[i].shape[-2], b_geo[i].shape[0]
            W = torch.nn.functional.interpolate(warp, size=(h, w), mode='bilinear').permute(0,2,3,1).repeat(b,1,1,1)
            b_geo[i] = torch.nn.functional.grid_sample(b_geo[i], W, align_corners=False)
            b_tex[i] = torch.nn.functional.grid_sample(b_tex[i], W, align_corners=False)

        ########################################
        if self.training:
            with torch.no_grad():
                w, h, b = neut_avgtex.shape[-1], neut_avgtex.shape[-2], neut_avgtex.shape[0]
                W = torch.nn.functional.interpolate(warp, size=(h, w), mode='bilinear').permute(0,2,3,1).repeat(b,1,1,1)
                tex = torch.nn.functional.grid_sample(neut_avgtex, W, align_corners=False)
                i1 = (neut_avgtex[0] * 10 + 150).clamp(0,255).permute(1,2,0).data.cpu().numpy().astype(np.uint8)
                i2 = (tex[0] * 10 + 150).clamp(0,255).permute(1,2,0).data.cpu().numpy().astype(np.uint8)


                wmin = torch.amin(self.bias)
                wmax = torch.amax(self.bias)
                W = torch.nn.functional.interpolate(self.bias, size=(h, w), mode='bilinear')
                W = (W[0].permute(1,2,0) + wmin) * 255 / (wmax - wmin + 1e-6)
                #W = (self.bias[0].permute(1,2,0).clamp(-0.01, 0.01) + 0.01) * 255. / 0.02
                i3 = torch.cat((W, W[:,:,:1]*0), dim=-1).data.cpu().numpy().astype(np.uint8)


                i4 = np.clip(np.abs(i1.astype(np.float32) - i2.astype(np.float32)) * 10, 0,255).astype(np.uint8)
                I = np.concatenate([i1,i2,i4,i3], axis=1)

                #from cv2 import imwrite
                #print(f'--------------------------------------------------------------------- {wmin*self.wsize} {wmax*self.wsize}')
                #imwrite('warp5.png', I)
        ########################################



        return {'z_geo': z_geo, 'z_tex': z_tex, 'b_geo': b_geo, 'b_tex': b_tex}, None


###############################################################################
class EncoderUNet(nn.Module):
    def __init__(self, ncond=1, imsize=1024, channel_mult=1, input_chan=3):
        super(EncoderUNet, self).__init__()

        self.ncond = ncond
        self.imsize = imsize
        l = models.utils.LinearWN
        c = models.utils.Conv2dWN
        #a = CenteredLeakyReLU
        a = nn.LeakyReLU
        s = nn.Sequential
        C = channel_mult

        if imsize == 1024:
            esize = [input_chan*ncond, 16*C, 32*C, 64*C, 64*C, 128*C, 128*C, 256*C, 256*C]
            bsize = [input_chan, 16, 32, 64, 64, 128, 128, 256, 256]
        else:
            print(f'Unsupported image size: {imsize}')
            quit()
        self.nlayers = len(esize)-1
        for i in range(self.nlayers):
            e = [c(esize[i], esize[i+1], 4, 2, 1)]
            b = [c(esize[i], bsize[i], 1, 1, 0)]
            e.append(a(0.2, inplace=True))
            if i > 0:
                b.append(a(0.2, inplace=True))
            self.add_module(f'e{i}', s(*e))
            self.add_module(f'b{i}', s(*b))
        self.enc = c(esize[-1], 16, 1, 1, 0)

        for i in range(self.nlayers):
            models.utils.initseq(self._modules[f'e{i}'])
            models.utils.initseq(self._modules[f'b{i}'])
        models.utils.initmod(self.enc)


    def forward(self, x):

        #############################
        x_orig = x
        #############################

        n, b = x.shape[0], []
        for i in range(self.nlayers):
            #skip first one since not used?
            #bi = None if i == 0 else self._modules[f'b{i}'](x)
            bi = self._modules[f'b{i}'](x)
            b.insert(0, bi)
            x = self._modules[f'e{i}'](x)
        z = self.enc(x)


        #######################################################################################
        if not np.isfinite(torch.sum(z).item()):
            print('------------------- Non Finite Encoding --------------------------')
            x = torch.sum(x_orig.contiguous().view(n, -1), dim=-1)
            print(f"x: {x}")

            z = torch.sum(z.contiguous().view(n, -1), dim=-1)
            print(f"z: {z}")

            for i in range(len(b)):
                bi = torch.sum(b[i].contiguous().view(n, -1), dim=-1)
                print(f"b{i}: {bi}")

            for i in range(len(b)):
                wi = torch.sum(self._modules[f'e{i}'][0].weight).item()
                bi = torch.sum(self._modules[f'e{i}'][0].bias).item()
                print(f"d{i}: {wi} {bi}")

            for i in range(len(b)):
                wi = torch.sum(self._modules[f'b{i}'][0].weight).item()
                bi = torch.sum(self._modules[f'b{i}'][0].weight).item()
                print(f"u{i}: {wi} {bi}")

            quit()
        #######################################################################################

        return z, b



###############################################################################
class GeoTexCombiner(nn.Module):
    def __init__(self, texsize=1024, geosize=1024, input_chan=3):
        super(GeoTexCombiner, self).__init__()

        self.texsize, self.geosize = texsize, geosize

        if self.texsize == 1024:
            tsize = [input_chan, 16, 32, 64, 64, 128, 128, 256]
        elif self.texsize == 512:
            tsize = [input_chan, 16, 32, 64, 64, 128, 128]
        elif self.texsize == 256:
            tsize = [input_chan, 16, 32, 64, 64, 128]
        elif self.texsize == 128:
            tsize = [input_chan, 16, 32, 64, 64]

        if self.geosize == 1024:
            gsize = [input_chan, 16, 32, 64, 64, 128, 128, 256]
        elif self.geosize == 512:
            gsize = [input_chan, 16, 32, 64, 64, 128, 128]
        elif self.geosize == 256:
            gsize = [input_chan, 16, 32, 64, 64, 128]
        elif self.geosize == 128:
            gsize = [input_chan, 16, 32, 64, 64]

        n = len(gsize)
        for i in range(n):
            sg, st = gsize[-1-i], tsize[-1-i]

            t2g = nn.Sequential(models.utils.Conv2dWN(st, sg, 1, 1, 0),
                                nn.LeakyReLU(0.2, inplace=True))
            g = nn.Sequential(models.utils.Conv2dWN(sg * 2, sg, 1, 1, 0),
                              nn.LeakyReLU(0.2, inplace=True))

            g2t = nn.Sequential(models.utils.Conv2dWN(sg, st, 1, 1, 0),
                                nn.LeakyReLU(0.2, inplace=True))
            t = nn.Sequential(models.utils.Conv2dWN(st * 2, st, 1, 1, 0),
                              nn.LeakyReLU(0.2, inplace=True))

            self.add_module(f't2g{i}', t2g)
            self.add_module(f'g2t{i}', g2t)
            self.add_module(f'g{i}', g)
            self.add_module(f't{i}', t)

            models.utils.initseq(self._modules[f'g{i}'])
            models.utils.initseq(self._modules[f't{i}'])
            models.utils.initseq(self._modules[f't2g{i}'])
            models.utils.initseq(self._modules[f'g2t{i}'])


    def forward(self, b_geo_id, b_tex_id):
        for i in range(len(b_geo_id)):
            cg = torch.cat([b_geo_id[i], self._modules[f't2g{i}'](b_tex_id[i])], dim=1)
            ct = torch.cat([b_tex_id[i], self._modules[f'g2t{i}'](b_geo_id[i])], dim=1)
            b_geo_id[i] = self._modules[f'g{i}'](cg)
            b_tex_id[i] = self._modules[f't{i}'](ct)

        return b_geo_id, b_tex_id
