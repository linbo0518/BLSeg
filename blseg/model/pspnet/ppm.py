import torch
from torch import nn
import torch.nn.functional as F
from ...backbone.utils import conv3x3


class PPM(nn.Module):

    def __init__(self, in_ch):
        super(PPM, self).__init__()
        out_ch = int(in_ch / 4)
        self.ppm1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.ppm2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.ppm3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.ppm4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = (x.size(2), x.size(3))
        ppm1 = F.interpolate(self.ppm1(x),
                             size=size,
                             mode='bilinear',
                             align_corners=True)
        ppm2 = F.interpolate(self.ppm2(x),
                             size=size,
                             mode='bilinear',
                             align_corners=True)
        ppm3 = F.interpolate(self.ppm3(x),
                             size=size,
                             mode='bilinear',
                             align_corners=True)
        ppm4 = F.interpolate(self.ppm4(x),
                             size=size,
                             mode='bilinear',
                             align_corners=True)
        ppm = torch.cat([x, ppm1, ppm2, ppm3, ppm4], dim=1)
        return ppm
