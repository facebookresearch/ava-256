import torch
import torch.nn as nn

# class Colorcal(nn.Module):
#     def __init__(self, allcameras):
#         super(Colorcal, self).__init__()

#         self.allcameras = allcameras

#         self.conv = nn.ModuleDict({
#             k: nn.Conv2d(3, 3, 1, 1, 0, groups=3) for k in self.allcameras})

#         for k in self.allcameras:
#             self.conv[k].weight.data[:] = 1.
#             self.conv[k].bias.data.zero_()

#     def forward(self, image, camindex):
#         return torch.cat([self.conv[self.allcameras[camindex[i].item()]](image[i:i+1, :, :, :]) for i in range(image.size(0))])

#     def parameters(self):
#         for p in super(Colorcal, self).parameters():
#             if p.requires_grad:
#                 yield p


class Colorcal(nn.Module):
    def __init__(self, ncams, nident, refcam = 0, refident = 0):
        super(Colorcal, self).__init__()

        self.ncams = ncams
        self.nident = nident
        
        wcam, bcam = [], []
        for i in range(ncams):
            wcam.append(nn.Parameter(torch.ones(3,)))
            bcam.append(nn.Parameter(torch.zeros(3,)))
        self.wcam = nn.ParameterList(wcam)
        self.bcam = nn.ParameterList(bcam)
        
        wident, bident = [], []
        for i in range(nident):
            wident.append(nn.Parameter(torch.zeros(3,)))
            bident.append(nn.Parameter(torch.zeros(3,)))    
        self.wident = nn.ParameterList(wident)
        self.bident = nn.ParameterList(bident)

        self.wcam[refcam].requires_grad = False
        self.bcam[refcam].requires_grad = False
        self.wident[refident].requires_grad = False
        self.bident[refident].requires_grad = False

    #@profile
    def forward(self, image, camindex, idindex):
        #cam = [self.allcameras.index(camindex[i].item()) for i in range(camindex.shape[0])]
        #ident = [self.allidentities.index(idindex[i].item()) for i in range(idindex.shape[0])]
        x = []
        for i in range(image.shape[0]):            
            w = self.wcam[camindex[i].item()] + self.wident[idindex[i].item()]
            b = self.bcam[camindex[i].item()] + self.bident[idindex[i].item()]
            x.append(w[None,:,None,None] * image[i:i+1,:,:,:] + \
                     b[None,:,None,None])
        return torch.cat(x, dim=0)



class Colorcal2(nn.Module):
    def __init__(self, ncams, nident, refcam = 0, refident = 0):
        super(Colorcal2, self).__init__()

        self.ncams = ncams
        self.nident = nident

        self.wcam = nn.Parameter(torch.ones(ncams, 3))
        self.bcam = nn.Parameter(torch.zeros(ncams, 3))
        self.wident = nn.Parameter(torch.zeros(nident, 3))
        self.bident = nn.Parameter(torch.zeros(nident, 3))
        
        #self.wcam[refcam].requires_grad = False
        #self.bcam[refcam].requires_grad = False
        #self.wident[refident].requires_grad = False
        #self.bident[refident].requires_grad = False

        self.refcam = refcam
        self.refident = refident

    #@profile
    def forward(self, image, camindex, idindex):#, **kwargs):
    #def forward(self, image=None, camindex=None, idindex=None, **kwargs):
        w = self.wcam[camindex] + self.wident[idindex]
        b = self.bcam[camindex] + self.bident[idindex]
        return w.unsqueeze(-1).unsqueeze(-1) * image + b.unsqueeze(-1).unsqueeze(-1)


#per identity and per camera model
class Colorcal3(nn.Module):
    def __init__(self, ncams, nident, refcam = 0, refident = 0):
        super(Colorcal3, self).__init__()

        self.ncams = ncams
        self.nident = nident

        self.w = nn.Parameter(torch.ones(ncams, nident, 3))
        self.b = nn.Parameter(torch.zeros(ncams, nident, 3))
        
        self.refcam = refcam
        self.refident = refident

    #@profile
    def forward(self, image, camindex, idindex):
        w = self.w[camindex, idindex]
        b = self.b[camindex, idindex]
        return w.unsqueeze(-1).unsqueeze(-1) * image + b.unsqueeze(-1).unsqueeze(-1)

#per identity and per camera model
#add a global component and increase learning rate for the specific cameras/identity pair
class Colorcal3_scaled(nn.Module):
    def __init__(self, ncams, nident, refcam = 0, refident = 0):
        super(Colorcal3_scaled, self).__init__()

        self.ncams = ncams
        self.nident = nident


        self.wcam = nn.Parameter(torch.ones(ncams, 3))
        self.bcam = nn.Parameter(torch.zeros(ncams, 3))
        self.wident = nn.Parameter(torch.zeros(nident, 3))
        self.bident = nn.Parameter(torch.zeros(nident, 3))
        self.w = nn.Parameter(torch.zeros(ncams, nident, 3))
        self.b = nn.Parameter(torch.zeros(ncams, nident, 3))
        
        self.refcam = refcam
        self.refident = refident

    #@profile
    def forward(self, image, camindex, idindex):
        w = self.wcam[camindex] + self.wident[idindex] + self.w[camindex, idindex] * 10
        b = self.bcam[camindex] + self.bident[idindex] + self.b[camindex, idindex] * 10
        return w.unsqueeze(-1).unsqueeze(-1) * image + b.unsqueeze(-1).unsqueeze(-1)


class Colorcal_TwoDatasets(nn.Module):
    def __init__(self, ncams1, nident1, ncams2, nident2):
        super(Colorcal_TwoDatasets, self).__init__()

        self.net1 = Colorcal2(ncams1, nident1)
        self.net2 = Colorcal2(ncams2, nident2)


    def forward(self, image, camindex, idindex, dataset_type):

        n = image.shape[0]
        I = []
        for i in range(n):
            if dataset_type[i] == 0:
                I.append(self.net1(image[i:i+1], camindex[i:i+1], idindex[i:i+1]))
            else:
                I.append(self.net2(image[i:i+1], camindex[i:i+1], idindex[i:i+1]))
        return torch.cat(I, dim=0)


#just normalize color
class ColorNorm(nn.Module):
    def __init__(self):
        super(ColorNorm, self).__init__()

    #@profile
    def forward(self, src, dst):

        if dst is None:
            return src

        b, h, w = src.shape[0], src.shape[-2], src.shape[-1]
        A = src.view(b,3,-1)
        B = dst.view(b,3,-1)

        #mean normalize
        Amean = torch.mean(A, dim=-1, keepdims=True)
        Bmean = torch.mean(B, dim=-1, keepdims=True)
        A = A - Amean
        B = B - Bmean
        AAt = torch.bmm(A, A.permute(0,2,1))
        BAt = torch.bmm(B, A.permute(0,2,1))
        for i in range(3):
            AAt[:,i,i] += 1e-3
        AAti = torch.inverse(AAt)
        x = torch.bmm(BAt, AAti)
        C = torch.bmm(x, A) + Bmean

        out = C.view(b,3,h,w)

        return out


#build running average of normalization
class ColorNorm2(nn.Module):
    def __init__(self):
        super(ColorNorm2, self).__init__()
        
        self.register_buffer("gain", torch.eye(3).unsqueeze(0))
        self.register_buffer("bias", torch.zeros(1,3,1))


    #@profile
    def forward(self, src, dst = None):

        b, h, w = src.shape[0], src.shape[-2], src.shape[-1]

        if dst is None:
            with torch.no_grad():
                out = torch.bmm(self.gain, src.view(b,3,-1)) + self.bias
            return out.view(b,3,h,w)

        A = src.view(b,3,-1)
        B = dst.view(b,3,-1)

        #mean normalize
        Amean = torch.mean(A, dim=-1, keepdims=True)
        Bmean = torch.mean(B, dim=-1, keepdims=True)
        A = A - Amean
        B = B - Bmean
        AAt = torch.bmm(A, A.permute(0,2,1))
        BAt = torch.bmm(B, A.permute(0,2,1))
        for i in range(3):
            AAt[:,i,i] += 1#1e-3
        AAti = torch.inverse(AAt)
        x = torch.bmm(BAt, AAti)
        C = torch.bmm(x, A) + Bmean

        out = C.view(b,3,h,w)

        with torch.no_grad():
            gain = x
            bias = Bmean - torch.bmm(x, Amean)

            alpha = 0.99
            self.gain *= alpha
            self.bias *= alpha
            self.gain += (1-alpha) * torch.mean(gain, dim=0, keepdims=True)
            self.bias += (1-alpha) * torch.mean(bias, dim=0, keepdims=True)

        return out

