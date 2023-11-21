import torch

class ColCatDataset(torch.nn.Module):
    def __init__(self, *args):
        super(ColCatDataset, self).__init__()

        self.datasets = args

    def __getattr__(self, attr):
        for x in self.datasets:
            try:
                return x.__getattribute__(attr)
            except:
                pass

        raise AttributeError("Can't find", attr, "on", x.__class__)

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        out = {}
        for d in self.datasets:
            out.update(d[idx])
        return out
        #return {**d[idx] for d in self.datasets}
