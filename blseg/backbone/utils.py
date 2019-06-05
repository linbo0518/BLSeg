import torch
from torch import nn


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


class DepthwiseSeparableConv(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 stride=1,
                 dilation=1,
                 relu6=True,
                 last_relu=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.dwconv = nn.Conv2d(in_ch,
                                in_ch,
                                3,
                                stride=stride,
                                padding=dilation,
                                dilation=dilation,
                                groups=in_ch,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if relu6:
            self.relu = nn.ReLU6(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.last_relu = last_relu

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pwconv(x)
        x = self.bn2(x)
        if self.last_relu:
            x = self.relu(x)
        return x


class Flatten(nn.Module):

    def __init__(self, start_dim=0, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return torch.flatten(x, self.start_dim, self.end_dim)
