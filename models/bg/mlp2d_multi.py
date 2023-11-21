import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.utils



class BackgroundModelSimple_TwoDatasets(nn.Module):
    def __init__(self, ncams, nident):
        super(BackgroundModelSimple_TwoDatasets, self).__init__()
        
        self.net = BackgroundModelSimple(ncams, nident)
        #self.net2 = BackgroundModelSimple(ncams, nident)


    def forward(self, dataset_type, camindex, idindex, samplecoords):
        
        n = camindex.shape[0]
        bg = []
        for i in range(n):
            im = self.net(camindex[i:i+1], idindex[i:i+1], samplecoords[i:i+1])
            if dataset_type[i] == 0:
                bg.append(im - 50)
            else:
                bg.append(im + 50)
        return torch.cat(bg, dim=0)
            
        #     bg.append(im)
        # return torch.cat(bg, dim=0)

        #bg = self.net(camindex, idindex, samplecoords)
        #v = (dataset_type.float() * 2 - 1)[:,None,None,None] * 50
        #return bg + v



class BackgroundModelSimple(nn.Module):
    def __init__(self, ncams, nident):
        super(BackgroundModelSimple, self).__init__()

        self.ncams = ncams
        self.nident = nident

        self.cammod = torch.nn.Sequential(
            nn.Linear(self.ncams, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 40)
        )
        self.idmod = torch.nn.Sequential(
            nn.Linear(self.nident, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 40)
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(40+40+40, 256, 1, 1, 0), torch.nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 1, 1, 0), torch.nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 1, 1, 0), torch.nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 1, 1, 0), torch.nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 1, 1, 0), torch.nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 3, 1, 1, 0))

        #self.apply(lambda x: la.glorot(x, 0.2))
        #la.glorot(self.mlp[-1], 1.0)

        initseq = models.utils.initseq
        initseq(self.cammod)
        initseq(self.idmod)
        initseq(self.mlp)

        cam_ident = torch.from_numpy(np.eye(self.ncams, dtype=np.float32))
        id_ident = torch.from_numpy(np.eye(self.nident, dtype=np.float32))
        self.register_buffer("cam_ident", cam_ident)
        self.register_buffer("id_ident", id_ident)

    def forward(self, camindex, idindex, samplecoords, **kwargs):

        cam = camindex
        ident = idindex

        b, h, w = samplecoords.shape[0], samplecoords.shape[1], samplecoords.shape[2]
        camenc = self.cammod(self.cam_ident[cam]).view(b,-1,1,1).repeat(1,1,h,w)
        idenc = self.idmod(self.id_ident[ident]).view(b,-1,1,1).repeat(1,1,h,w)
        posenc = torch.cat([torch.sin(2 ** i * np.pi * samplecoords) for i in range(10)] + 
                           [torch.cos(2 ** i * np.pi * samplecoords) for i in range(10)], 
                           dim=-1).permute(0, 3, 1, 2)
        decout = self.mlp(torch.cat([camenc, idenc, posenc], dim=1))
        bg = decout * 25. + 100.
        
        return bg



class BGModel_TwoDatasets(nn.Module):
    def __init__(self, nident, width, height, allcameras, bgdict=True, trainstart=0):
        super(BGModel_TwoDatasets, self).__init__()
        self.net = BGModel(nident, width, height, allcameras, bgdict, trainstart)

    def forward(self, dataset_type, bg=None, camindex=None, idindex=None, raypos=None, rayposend=None, raydir=None, samplecoords=None, trainiter=-1, **kwargs):
        
        return self.net(bg, camindex, idindex, raypos, rayposend, raydir, samplecoords, trainiter)

        # bg = self.net(bg, camindex, idindex, raypos, rayposend, raydir, samplecoords, trainiter)
        # v = (dataset_type.float() * 2 - 1)[:,None,None,None] * 50
        # return bg + v
        

class BGModel(nn.Module):
    def __init__(self, nident, width, height, allcameras, bgdict=True, trainstart=0):
        super(BGModel, self).__init__()

        self.nident = nident
        self.allcameras = allcameras
        self.trainstart = trainstart

        if bgdict:
            self.bg = models.utils.BufferDict({k: torch.ones(3, height, width) for k in allcameras})
        else:
            self.bg = None


        self.mlp0 = nn.Sequential(
                nn.Linear(nident, 256), nn.LeakyReLU(0.2),
                nn.Linear(  256, 64))

        self.mlp1 = nn.Sequential(
                nn.Conv2d(60+24+64, 256, 1, 1, 0), nn.LeakyReLU(0.2),
                nn.Conv2d(  256, 256, 1, 1, 0), nn.LeakyReLU(0.2),
                nn.Conv2d(  256, 256, 1, 1, 0), nn.LeakyReLU(0.2),
                nn.Conv2d(  256, 256, 1, 1, 0), nn.LeakyReLU(0.2),
                nn.Conv2d(  256, 256, 1, 1, 0))

        self.mlp2 = nn.Sequential(
                nn.Conv2d(60+24+64+256, 256, 1, 1, 0), nn.LeakyReLU(0.2),
                nn.Conv2d(         256, 256, 1, 1, 0), nn.LeakyReLU(0.2),
                nn.Conv2d(         256, 256, 1, 1, 0), nn.LeakyReLU(0.2),
                nn.Conv2d(         256,   3, 1, 1, 0))

        initseq = models.utils.initseq
        for m in [self.mlp1, self.mlp2]:
            initseq(m)

        
        self.mlp0[0].weight.data.normal_(0, 1)
        self.mlp0[0].bias.data.zero_()

        onehot = torch.from_numpy(np.eye(self.nident, dtype=np.float32))
        self.register_buffer("onehot", onehot)


    #@profile
    def forward(self, bg=None, camindex=None, idindex=None, raypos=None, rayposend=None, raydir=None, samplecoords=None, trainiter=-1, **kwargs):
        # generate position encoding
        #posenc = torch.cat([
        #    torch.sin(2 ** i * np.pi * rayposend[:, :, :, :])
        #    for i in range(10)] + [
        #    torch.cos(2 ** i * np.pi * rayposend[:, :, :, :])
        #    for i in range(10)], dim=-1).permute(0, 3, 1, 2)

        #direnc = torch.cat([
        #    torch.sin(2 ** i * np.pi * raydir[:, :, :, :])
        #    for i in range(4)] + [
        #    torch.cos(2 ** i * np.pi * raydir[:, :, :, :])
        #    for i in range(4)], dim=-1).permute(0, 3, 1, 2)

        #decout = self.mlp1(torch.cat([
        #    posenc,
        #    direnc], dim=1))

        #decout = self.mlp2(torch.cat([posenc, direnc, decout], dim=1))

        if trainiter >= self.trainstart and camindex is not None and idindex is not None:
            posenc = torch.cat([
                torch.sin(2 ** i * np.pi * rayposend[:, :, :, :])
                for i in range(10)] + [
                torch.cos(2 ** i * np.pi * rayposend[:, :, :, :])
                for i in range(10)], dim=-1).permute(0, 3, 1, 2)

            direnc = torch.cat([
                torch.sin(2 ** i * np.pi * raydir[:, :, :, :])
                for i in range(4)] + [
                torch.cos(2 ** i * np.pi * raydir[:, :, :, :])
                for i in range(4)], dim=-1).permute(0, 3, 1, 2)

            h, w = posenc.shape[-2], posenc.shape[-1]
            #onehot = torch.zeros(idindex.shape[0], self.nident)
            #for i in range(onehot.shape[0]):
            #    onehot[i, idindex[i].item()] = 1
            #onehot = onehot.to('cuda').requires_grad_(False)
            #onehot = torch.nn.functional.one_hot(idindex, num_classes=self.nident).float().to('cuda').requires_grad_(False)
            onehot = self.onehot[idindex]
            idenc = self.mlp0(onehot).unsqueeze(-1).unsqueeze(-1).repeat(1,1,h,w)
            
            decout = self.mlp1(torch.cat([
                posenc,
                direnc,
                idenc], dim=1))

            decout = self.mlp2(torch.cat([posenc, direnc, idenc, decout], dim=1))
        else:
            decout = None


        if bg is None and self.bg is not None and camindex is not None:
            bg = torch.stack([self.bg[self.allcameras[camindex[i].item()]] for i in range(camindex.size(0))], dim=0)
        else:
            bg = None

        if bg is not None and samplecoords is not None:
            # inverse softplus
            #bg = torch.where(bg > 15., bg, torch.log((torch.exp(bg) - 1.).clamp(min=1e-6)))

            if samplecoords.size()[1:3] != bg.size()[2:4]:
                bg = F.grid_sample(bg, samplecoords)

        #if trainiter >= self.trainstart:
        #    print("here")
        #    return F.softplus(bg + decout)
        #else:
        #    return F.softplus(bg + decout * 0.) # include in graph?

        if decout is not None:
            if bg is not None:
                return F.softplus(bg + decout)
            else:
                return F.softplus(decout)
        else:
            if bg is not None:
                return F.softplus(bg)
            else:
                return None
