import torch
from torch import nn
import torch.nn.functional as F
from ...backbone.utils import conv3x3
from ..base import SegBaseModule
from .aspp import ASPP


class DeepLabV3Plus(SegBaseModule):

    def __init__(self,
                 backbone='xception',
                 num_classes=21,
                 dilations=[1, 1, 1, 1, 2]):
        assert backbone in [
            'vgg16', 'resnet34', 'resnet50', 'se_resnet34', 'se_resnet50',
            'mobilenet_v1', 'mobilenet_v2', 'xception'
        ]
        super(DeepLabV3Plus, self).__init__(num_classes)
        self.backbone = self._get_backbone(backbone)
        self.backbone.change_output_stride(16)
        self.backbone.change_dilation(dilations)
        self.aspp = ASPP(self.backbone.channels[4])
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(self.backbone.channels[1], 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.concat_conv = nn.Sequential(
            conv3x3(304, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1, bias=False),
        )

        self._init_params()

    def forward(self, x):
        x = self.backbone.stage0(x)
        low_level_features = self.backbone.stage1(x)
        x = self.backbone.stage2(low_level_features)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)
        aspp_out = self.aspp(x)
        aspp_out = F.interpolate(aspp_out,
                                 scale_factor=4,
                                 mode='bilinear',
                                 align_corners=True)
        low_level_features = self.low_level_conv(low_level_features)
        out = torch.cat((aspp_out, low_level_features), dim=1)
        out = self.concat_conv(out)
        out = F.interpolate(out,
                            scale_factor=4,
                            mode='bilinear',
                            align_corners=True)
        return out

    def reset_classes(self, num_classes):
        self.num_classes = num_classes
        self.concat_conv[-1] = nn.Conv2d(256, num_classes, 1, bias=False)