import torch
from torch import nn
from .utils import conv3x3
from .base import BackboneBaseModule


class BasicBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride, expansion=1):
        assert out_ch % expansion == 0
        mid_ch = int(out_ch / expansion)
        super(BasicBlock, self).__init__()
        self.do_downsample = not (in_ch == out_ch and stride == 1)
        self.conv1 = conv3x3(in_ch, mid_ch, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = conv3x3(mid_ch, out_ch, stride=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if self.do_downsample:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.do_downsample:
            residual = self.residual(residual)
        x += residual
        return self.relu(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride, expansion=4):
        assert out_ch % expansion == 0
        mid_ch = int(out_ch / expansion)
        super(ResidualBlock, self).__init__()
        self.do_downsample = not (in_ch == out_ch and stride == 1)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = conv3x3(mid_ch, mid_ch, stride)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if self.do_downsample:
            self.residual = nn.Sequential(
                nn.AvgPool2d(stride, stride, ceil_mode=True),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.do_downsample:
            residual = self.residual(residual)
        x += residual
        return self.relu(x)


class ResNet34(BackboneBaseModule):

    def __init__(self):
        super(ResNet34, self).__init__()
        self.channels = [64, 64, 128, 256, 512]
        self.strides = [2, 4, 8, 16, 32]
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, self.channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(nn.MaxPool2d(3, stride=2, padding=1))
        for layer in self._add_stage(BasicBlock, self.channels[0],
                                     self.channels[1], 1, 3):
            self.stage1.add_module(str(len(self.stage1)), layer)
        self.stage2 = self._add_stage(BasicBlock, self.channels[1],
                                      self.channels[2], 2, 4)
        self.stage3 = self._add_stage(BasicBlock, self.channels[2],
                                      self.channels[3], 2, 6)
        self.stage4 = self._add_stage(BasicBlock, self.channels[3],
                                      self.channels[4], 2, 3)

        self._init_params()

    def forward(self, x):
        x = self.stage0(x)  # 64, 1/2
        x = self.stage1(x)  # 64, 1/4
        x = self.stage2(x)  # 128, 1/8
        x = self.stage3(x)  # 256, 1/16
        x = self.stage4(x)  # 512, 1/32
        return x

    def _add_stage(self, block, in_ch, out_ch, stride, repeat_time):
        assert repeat_time > 0 and isinstance(repeat_time, int)
        layers = [block(in_ch, out_ch, stride)]
        for _ in range(repeat_time - 1):
            layers.append(block(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _change_downsample(self, params):
        self.stage3[0].conv1.stride = (params[0], params[0])
        self.stage3[0].residual[0].stride = params[0]
        self.stage4[0].conv1.stride = (params[1], params[1])
        self.stage4[0].residual[0].stride = params[1]


class ResNet50S(BackboneBaseModule):

    def __init__(self):
        super(ResNet50S, self).__init__()
        self.channels = [64, 256, 512, 1024, 2048]
        self.strides = [2, 4, 8, 16, 32]
        self.stage0 = nn.Sequential(
            conv3x3(3, 32, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv3x3(32, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv3x3(32, self.channels[0]),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(nn.MaxPool2d(3, stride=2, padding=1))
        for layer in self._add_stage(ResidualBlock, self.channels[0],
                                     self.channels[1], 1, 3):
            self.stage1.add_module(str(len(self.stage1)), layer)
        self.stage2 = self._add_stage(ResidualBlock, self.channels[1],
                                      self.channels[2], 2, 4)
        self.stage3 = self._add_stage(ResidualBlock, self.channels[2],
                                      self.channels[3], 2, 6)
        self.stage4 = self._add_stage(ResidualBlock, self.channels[3],
                                      self.channels[4], 2, 3)

        self._init_params()

    def forward(self, x):
        x = self.stage0(x)  # 64, 1/2
        x = self.stage1(x)  # 256, 1/4
        x = self.stage2(x)  # 512, 1/8
        x = self.stage3(x)  # 1024, 1/16
        x = self.stage4(x)  # 2048, 1/32
        return x

    def _add_stage(self, block, in_ch, out_ch, stride, repeat_time):
        assert repeat_time > 0 and isinstance(repeat_time, int)
        layers = [block(in_ch, out_ch, stride)]
        for _ in range(repeat_time - 1):
            layers.append(block(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _change_downsample(self, params):
        self.stage3[0].conv2.stride = (params[0], params[0])
        self.stage3[0].residual[0].kernel_size = params[0]
        self.stage3[0].residual[0].stride = params[0]
        self.stage4[0].conv2.stride = (params[1], params[1])
        self.stage4[0].residual[0].kernel_size = params[1]
        self.stage4[0].residual[0].stride = params[1]
