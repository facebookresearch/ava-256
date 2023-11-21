import torch
import torch.nn as nn

class Colorcal(nn.Module):
    def __init__(self, allcameras):
        super(Colorcal, self).__init__()

        self.allcameras = allcameras

        self.conv = nn.ModuleDict({
            k: nn.Conv2d(3, 3, 1, 1, 0, groups=3) for k in self.allcameras})

        for k in self.allcameras:
            self.conv[k].weight.data[:] = 1.
            self.conv[k].bias.data.zero_()

    def forward(self, image, camindex):
        return torch.cat([self.conv[self.allcameras[camindex[i].item()]](image[i:i+1, :, :, :]) for i in range(image.size(0))])

    def parameters(self):
        for p in super(Colorcal, self).parameters():
            if p.requires_grad:
                yield p
