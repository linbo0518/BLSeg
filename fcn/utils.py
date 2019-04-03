import torch
from torch import nn


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride):
        super(ResidualBlock, self).__init__()
        self.do_downsample = not (in_ch == out_ch and stride == 1)
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if self.do_downsample:
            self.conv3 = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
            self.bn3 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.do_downsample:
            residual = self.conv3(residual)
            residual = self.bn3(residual)

        x += residual
        x = self.relu(x)
        return x
