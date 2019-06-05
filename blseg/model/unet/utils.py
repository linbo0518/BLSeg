import torch
from torch import nn
from torch.nn import functional as F
from ...backbone.utils import conv3x3
from ...backbone.resnet import BasicBlock


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
        self.interpolate_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 2, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=self.scale_factor,
                          mode='bilinear',
                          align_corners=True)
        x = F.pad(x, (0, 1, 0, 1))
        return self.interpolate_conv(x)


class UpBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.up_conv = UpConv(in_ch, out_ch)
        self.up_block = nn.Sequential(
            conv3x3(out_ch * 2, out_ch),
            nn.ReLU(inplace=True),
            conv3x3(out_ch, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, encoded, x):
        x = self.up_conv(x)
        x = torch.cat((encoded, x), dim=1)
        return self.up_block(x)


class ModernUpConv(nn.Module):

    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(ModernUpConv, self).__init__()
        self.scale_factor = scale_factor
        self.interpolate_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=self.scale_factor,
                          mode='bilinear',
                          align_corners=True)
        return self.interpolate_conv(x)


class ModernUpBlock(nn.Module):

    def __init__(self, in_ch, out_ch, use_se=False):
        super(ModernUpBlock, self).__init__()
        self.up_conv = ModernUpConv(in_ch, out_ch)
        self.up_block = BasicBlock(out_ch * 2, out_ch, 1, use_se=use_se)

    def forward(self, encoded, x):
        x = self.up_conv(x)
        x = torch.cat((encoded, x), dim=1)
        return self.up_block(x)