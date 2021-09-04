import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_chans=None, out_chans=None, stride=1):
        super(BasicBlock,self).__init__()
        lst = []
        self.shortcut = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=1, stride=stride)
        self.cnv0 = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, stride=stride, padding=1)
        self.bn0 = nn.BatchNorm2d(num_features=out_chans)
        self.cnv1 = nn.Conv2d(in_channels=out_chans, out_channels=out_chans, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_chans)

    def forward(self, X):
        xs = self.shortcut(X)
        X = self.cnv0(X)
        X = self.bn0(X)
        X = F.relu(X)
        X = self.cnv1(X)
        X = self.bn1(X)

        Y = F.relu(X+xs)
        return Y

class Layer(nn.Module):
    def __init__(self, in_chans=None, out_chans=None, nblocks=5):
        super(Layer,self).__init__()
        mod = []
        mod.append(BasicBlock(in_chans=in_chans,out_chans=out_chans, stride=2))
        for k in range(1, nblocks):
            mod.append(BasicBlock(in_chans=out_chans,out_chans=out_chans,stride=1))
        self.models = nn.ModuleList(mod)

    def forward(self, X):
        for m in self.models:
            X = m(X)
        return X

class ResNet50(nn.Module):
    def __init__(self, xsize=32, ysize=32, nfliters=64, nhid=512, ncls=10, softmask_flg=True):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, nfliters, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(nfliters)
        min_sz = 4
        mod = []
        out_chans = nfliters
        self.softmask_flg = softmask_flg
        while xsize>min_sz and ysize>min_sz:
            in_chans = out_chans
            nfliters *= 2
            out_chans = nfliters
            mod.append(Layer(in_chans=in_chans,out_chans=out_chans))
            xsize /= 2
            ysize /= 2
        nfeatures = int(out_chans*xsize * ysize)
        mod.append(nn.Flatten())
        self.lin1 = nn.Linear(in_features=nfeatures, out_features=nhid)
        self.lin2 = nn.Linear(in_features=nhid, out_features=ncls)
        self.lays = nn.ModuleList(mod)
    def forward(self,X):
        X = self.conv1(X)
        X = self.bn1(X)
        for m in self.lays:
            X = m (X)
        X = self.lin1(X)
        X = F.relu(X)
        X = self.lin2(X)
        if self.softmask_flg:
            Y = F.softmax(X, dim=1)
        else:
            Y = X

        return Y


if __name__ == "__main__":
    rnet = ResNet50()
    X = torch.randn((3,3,32,32))
    Y = rnet(X)
    print(rnet)