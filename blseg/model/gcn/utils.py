import torch
from torch import nn
from ...backbone.utils import conv3x3


class GCNModule(nn.Module):

    def __init__(self, in_ch, out_ch, k):
        super(GCNModule, self).__init__()
        padding = (k - 1) // 2
        self.gcn1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (k, 1), padding=(padding, 0), bias=False),
            nn.Conv2d(out_ch, out_ch, (1, k), padding=(0, padding), bias=False),
        )
        self.gcn2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, k), padding=(0, padding), bias=False),
            nn.Conv2d(out_ch, out_ch, (k, 1), padding=(padding, 0), bias=False),
        )

    def forward(self, x):
        gcn1 = self.gcn1(x)
        gcn2 = self.gcn2(x)
        return gcn1 + gcn2


class BRModule(nn.Module):

    def __init__(self, in_ch):
        super(BRModule, self).__init__()
        self.br = nn.Sequential(
            conv3x3(in_ch, in_ch),
            nn.ReLU(inplace=True),
            conv3x3(in_ch, in_ch),
        )

    def forward(self, x):
        br = self.br(x)
        return x + br


class Pipeline(nn.Module):

    def __init__(self, in_ch, out_ch, k):
        super(Pipeline, self).__init__()
        self.gcn = GCNModule(in_ch, out_ch, k)
        self.br1 = BRModule(out_ch)
        self.br2 = BRModule(out_ch)
        self.deconv = nn.ConvTranspose2d(out_ch, out_ch, 2, 2, bias=False)

    def forward(self, x, fuse):
        x = self.gcn(x)
        x = self.br1(x)
        x = self.br2(x + fuse)
        x = self.deconv(x)
        return x
