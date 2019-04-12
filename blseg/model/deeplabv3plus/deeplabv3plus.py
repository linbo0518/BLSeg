import torch
from torch import nn
import torch.nn.functional as F
from ..utils import _get_backbone
from .aspp import ASPP


class DeepLabV3Plus(nn.Module):

    def __init__(self, backbone='xception', num_classes=1):
        assert backbone in ['vgg16', 'resnet50', 'mobilenetv1', 'xception']
        super(DeepLabV3Plus, self).__init__()
        self.backbone = _get_backbone(backbone)
        self.backbone.change_output_stride(16)
        self.aspp = ASPP(self.backbone.channels[4])
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(self.backbone.channels[1], 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.concat_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self._init_params()

    def forward(self, x):
        x = self.backbone.stage0(x)
        low_level_features = self.backbone.stage1(x)
        x = self.backbone.stage2(low_level_features)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)
        aspp_out = self.aspp(x)
        low_level_features = self.low_level_conv(low_level_features)
        aspp_out = F.interpolate(aspp_out,
                                 scale_factor=4,
                                 mode='bilinear',
                                 align_corners=False)
        out = torch.cat((aspp_out, low_level_features), dim=1)
        out = self.concat_conv(out)
        out = F.interpolate(out,
                            scale_factor=4,
                            mode='bilinear',
                            align_corners=False)
        return out

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)