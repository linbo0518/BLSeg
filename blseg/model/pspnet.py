import torch
from torch import nn
import torch.nn.functional as F
from ..backbone.utils import conv3x3
from .base import SegBaseModule

__all__ = ["PSPNet"]


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


class PSPNet(SegBaseModule):
    def __init__(self,
                 backbone='resnet50',
                 num_classes=21,
                 dilations=(1, 1, 1, 2, 4)):
        assert backbone in [
            'vgg16', 'resnet34', 'resnet50', 'se_resnet34', 'se_resnet50',
            'mobilenet_v1', 'mobilenet_v2', 'xception'
        ]
        super(PSPNet, self).__init__(num_classes)
        self.backbone = self._get_backbone(backbone)
        self.backbone.change_output_stride(8)
        self.backbone.change_dilation(dilations)
        self.ppm = PPM(self.backbone.channels[4])
        self.ppm_conv = nn.Sequential(
            conv3x3(self.backbone.channels[4] * 2, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.out_conv = nn.Conv2d(512, num_classes, 1)

        self._init_params()

    def forward(self, x):
        x = self.backbone.stage0(x)
        x = self.backbone.stage1(x)
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)
        x = self.ppm(x)
        x = self.ppm_conv(x)
        x = self.out_conv(x)
        out = F.interpolate(x,
                            scale_factor=8,
                            mode='bilinear',
                            align_corners=True)
        return out

    def reset_classes(self, num_classes):
        self.num_classes = num_classes
        self.out_conv = nn.Conv2d(512, num_classes, 1)