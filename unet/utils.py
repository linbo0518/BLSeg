import torch
from torch import nn
import torch.nn.functional as F


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


class DownBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.down_block = nn.Sequential(
            nn.MaxPool2d(2, 2),
            conv3x3(in_ch, out_ch),
            nn.ReLU(inplace=True),
            conv3x3(out_ch, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.down_block(x)


class UpConv(nn.Module):

    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(UpConv, self).__init__()
        self.scale_factor = scale_factor
        self.conv = conv3x3(in_ch, out_ch)

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=self.scale_factor,
                          mode='bilinear',
                          align_corners=False)
        return self.conv(x)


class UpBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.up_conv = UpConv(in_ch, out_ch)
        self.up_block = nn.Sequential(
            conv3x3(in_ch, out_ch),
            nn.ReLU(inplace=True),
            conv3x3(out_ch, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, encoded, x):
        x = self.up_conv(x)
        x = torch.cat((encoded, x), dim=1)
        return self.up_block(x)


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


class DecodeBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DecodeBlock, self).__init__()
        self.up_conv = UpConv(in_ch, out_ch)
        self.decode_block = ResidualBlock(in_ch, out_ch, 1)

    def forward(self, encoded, x):
        x = self.up_conv(x)
        x = torch.cat((encoded, x), dim=1)
        return self.decode_block(x)
