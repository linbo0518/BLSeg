import torch
from torch import nn
import torch.nn.functional as F
from ..base import SegBaseModule
from .ppm import PPM


class PSPNet(SegBaseModule):

    def __init__(self, backbone='resnet50', num_classes=1):
        assert backbone in ['vgg16', 'resnet50', 'mobilenetv1', 'xception']
        super(PSPNet, self).__init__()
        self.backbone = self._get_backbone(backbone)
        self.backbone.change_output_stride(8)
        self.backbone.change_dilation([1, 1, 1, 2, 4])
        self.ppm = PPM(self.backbone.channels[4])
        self.final_conv = nn.Conv2d(int(self.backbone.channels[4] / 4),
                                    num_classes,
                                    1,
                                    bias=False)

        self._init_params()

    def forward(self, x):
        x = self.backbone.stage0(x)
        x = self.backbone.stage1(x)
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)
        x = self.ppm(x)
        x = self.final_conv(x)
        out = F.interpolate(x,
                            scale_factor=8,
                            mode='bilinear',
                            align_corners=False)
        return out
