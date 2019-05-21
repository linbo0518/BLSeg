import torch
from torch import nn
import torch.nn.functional as F


class ASPP(nn.Module):

    def __init__(self, in_ch):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp_out = nn.Sequential(
            nn.Conv2d(1280, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = (x.size(2), x.size(3))
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        aspp5 = F.interpolate(self.aspp5(x),
                              size=size,
                              mode='bilinear',
                              align_corners=True)
        all_aspp = torch.cat((aspp1, aspp2, aspp3, aspp4, aspp5), dim=1)
        return self.aspp_out(all_aspp)
