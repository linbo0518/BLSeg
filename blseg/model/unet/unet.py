import torch
from torch import nn
from ...backbone.utils import conv3x3
from ..base import SegBaseModule
from .utils import DownBlock, UpBlock, UpConv, ModernUpBlock, ModernUpConv


class UNet(SegBaseModule):

    def __init__(self, num_classes=1):
        super(UNet, self).__init__(num_classes)
        self.inputs = nn.Sequential(
            conv3x3(3, 64),
            nn.ReLU(inplace=True),
            conv3x3(64, 64),
            nn.ReLU(inplace=True),
        )
        self.down_block1 = DownBlock(64, 128)
        self.down_block2 = DownBlock(128, 256)
        self.down_block3 = DownBlock(256, 512)
        self.down_block4 = DownBlock(512, 1024)
        self.up_block4 = UpBlock(1024, 512)
        self.up_block3 = UpBlock(512, 256)
        self.up_block2 = UpBlock(256, 128)
        self.up_block1 = UpBlock(128, 64)
        self.outputs = nn.Conv2d(64, num_classes, 1, bias=False)

        self._init_params()

    def forward(self, x):
        e1 = self.inputs(x)
        e2 = self.down_block1(e1)
        e3 = self.down_block2(e2)
        e4 = self.down_block3(e3)
        e5 = self.down_block4(e4)
        d4 = self.up_block4(e4, e5)
        d3 = self.up_block3(e3, d4)
        d2 = self.up_block2(e2, d3)
        d1 = self.up_block1(e1, d2)
        return self.outputs(d1)


class ModernUNet(SegBaseModule):

    def __init__(self, backbone='resnet50', num_classes=1):
        assert backbone in [
            'vgg16', 'resnet50', 'mobilenetv1', 'mobilenetv2', 'xception'
        ]
        super(ModernUNet, self).__init__(num_classes)
        self.backbone = self._get_backbone(backbone)

        self.up_block4 = ModernUpBlock(self.backbone.channels[4],
                                       self.backbone.channels[3])
        self.up_block3 = ModernUpBlock(self.backbone.channels[3],
                                       self.backbone.channels[2])
        self.up_block2 = ModernUpBlock(self.backbone.channels[2],
                                       self.backbone.channels[1])
        self.up_block1 = ModernUpBlock(self.backbone.channels[1],
                                       self.backbone.channels[0])
        self.outputs = nn.Sequential(
            ModernUpConv(self.backbone.channels[0], self.backbone.channels[0]),
            nn.Conv2d(self.backbone.channels[0], num_classes, 1, bias=False),
        )

        self._init_params()

    def forward(self, x):
        e1 = self.backbone.stage0(x)
        e2 = self.backbone.stage1(e1)
        e3 = self.backbone.stage2(e2)
        e4 = self.backbone.stage3(e3)
        e5 = self.backbone.stage4(e4)
        d4 = self.up_block4(e4, e5)
        d3 = self.up_block3(e3, d4)
        d2 = self.up_block2(e2, d3)
        d1 = self.up_block1(e1, d2)
        return self.outputs(d1)
