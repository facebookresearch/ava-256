import numpy as np
import torch
import torch.nn as nn

import models.utils


class BackgroundModelSimple(nn.Module):
    """A very simple background model with modifiers for each camera and identity, as positional encoding for pixels"""

    def __init__(self, ncams, nident):
        super(BackgroundModelSimple, self).__init__()

        self.ncams = ncams
        self.nident = nident

        self.cammod = torch.nn.Sequential(
            nn.Linear(self.ncams, 256), torch.nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, 40)
        )
        self.idmod = torch.nn.Sequential(
            nn.Linear(self.nident, 256), torch.nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, 40)
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(40 + 40 + 40, 256, 1, 1, 0),
            torch.nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 1, 1, 0),
            torch.nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 1, 1, 0),
            torch.nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 1, 1, 0),
            torch.nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 1, 1, 0),
            torch.nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 3, 1, 1, 0),
        )

        initseq = models.utils.initseq
        initseq(self.cammod)
        initseq(self.idmod)
        initseq(self.mlp)

        # One-hot encoders for camera and identity
        # TODO(julieta) see if we can replace this indexing with torch's functional F.one_hot
        cam_ident = torch.from_numpy(np.eye(self.ncams, dtype=np.float32))
        id_ident = torch.from_numpy(np.eye(self.nident, dtype=np.float32))
        self.register_buffer("cam_ident", cam_ident)
        self.register_buffer("id_ident", id_ident)

    def forward(
        self, camindex: torch.LongTensor, idindex: torch.LongTensor, samplecoords: torch.FloatTensor
    ) -> torch.Tensor:
        b, h, w = samplecoords.shape[0], samplecoords.shape[1], samplecoords.shape[2]

        camenc = self.cammod(self.cam_ident[camindex]).view(b, -1, 1, 1).repeat(1, 1, h, w)
        idenc = self.idmod(self.id_ident[idindex]).view(b, -1, 1, 1).repeat(1, 1, h, w)

        posenc = torch.cat(
            [torch.sin(2**i * np.pi * samplecoords) for i in range(10)]
            + [torch.cos(2**i * np.pi * samplecoords) for i in range(10)],
            dim=-1,
        ).permute(0, 3, 1, 2)
        decout = self.mlp(torch.cat([camenc, idenc, posenc], dim=1))
        bg = decout * 25.0 + 100.0

        return bg
