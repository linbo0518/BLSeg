import torch
from torch import nn
from ..backbone import VGG16, MobileNetV1, ResNet50S
from ..backbone.utils import conv3x3
from .utils import DownBlock, UpBlock, UpConv


class UNet(nn.Module):

    def __init__(self, num_classes=1):
        super(UNet, self).__init__()
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

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')


class ModernUNet(nn.Module):

    def __init__(self, backbone='resnet50', num_classes=1):
        assert backbone in ['vgg16', 'resnet50', 'mobilenetv1']
        super(ModernUNet, self).__init__()
        if backbone == 'vgg16':
            self.backbone = VGG16()
        elif backbone == 'resnet50':
            self.backbone = ResNet50S()
        elif backbone == 'mobilenetv1':
            self.backbone = MobileNetV1()

        self.up_block4 = UpBlock(self.backbone.channels[4],
                                 self.backbone.channels[3])
        self.up_block3 = UpBlock(self.backbone.channels[3],
                                 self.backbone.channels[2])
        self.up_block2 = UpBlock(self.backbone.channels[2],
                                 self.backbone.channels[1])
        self.up_block1 = UpBlock(self.backbone.channels[1],
                                 self.backbone.channels[0])
        self.outputs = nn.Sequential(
            UpConv(self.backbone.channels[0], self.backbone.channels[0]),
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

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)