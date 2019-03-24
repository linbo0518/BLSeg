import torch
from torch import nn
import torch.nn.functional as F
from utils import conv3x3, DownBlock, UpBlock, ResidualBlock, DecodeBlock
from backbone import ResNet34


class UNet(nn.Module):

    def __init__(self, in_ch=3, num_classes=1):
        super(UNet, self).__init__()
        self.inputs = nn.Sequential(
            conv3x3(in_ch, 64),
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

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


class ResUNet(nn.Module):

    def __init__(self, num_classes=1):
        super(ResUNet, self).__init__()
        self.encoder = ResNet34()
        self.center = ResidualBlock(512, 1024, 2)
        self.decoder4 = DecodeBlock(1024, 512)
        self.decoder3 = DecodeBlock(512, 256)
        self.decoder2 = DecodeBlock(256, 128)
        self.decoder1 = DecodeBlock(128, 64)
        self.outputs = nn.Conv2d(64, num_classes, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e0 = self.encoder.stage0(x)
        e1 = self.encoder.stage1(e0)
        e2 = self.encoder.stage2(e1)
        e3 = self.encoder.stage3(e2)
        e4 = self.encoder.stage4(e3)
        e5 = self.center(e4)
        d4 = self.decoder4(e4, e5)
        d3 = self.decoder3(e3, d4)
        d2 = self.decoder2(e2, d3)
        d1 = self.decoder1(e1, d2)
        d1 = F.interpolate(
            d1, scale_factor=2, mode='bilinear', align_corners=False)
        return self.outputs(d1)
