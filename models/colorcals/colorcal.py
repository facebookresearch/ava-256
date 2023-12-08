import torch
import torch.nn as nn


class Colorcal(nn.Module):
    def __init__(self, ncams, nident, refcam=0, refident=0):
        super(Colorcal, self).__init__()

        self.ncams = ncams
        self.nident = nident

        self.wcam = nn.Parameter(torch.ones(ncams, 3))
        self.bcam = nn.Parameter(torch.zeros(ncams, 3))
        self.wident = nn.Parameter(torch.zeros(nident, 3))
        self.bident = nn.Parameter(torch.zeros(nident, 3))

        self.refcam = refcam
        self.refident = refident

    def forward(self, image, camindex, idindex):
        w = self.wcam[camindex] + self.wident[idindex]
        b = self.bcam[camindex] + self.bident[idindex]
        return w.unsqueeze(-1).unsqueeze(-1) * image + b.unsqueeze(-1).unsqueeze(-1)
