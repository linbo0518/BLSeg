import torch
from torch import nn
from .utils import conv3x3, DepthwiseSeparableConv, ResidualBlock


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.channels = [64, 128, 256, 512, 512]
        self.stage0 = self._add_stage(3, self.channels[0], 2)
        self.stage1 = self._add_stage(self.channels[0], self.channels[1], 2)
        self.stage2 = self._add_stage(self.channels[1], self.channels[2], 3)
        self.stage3 = self._add_stage(self.channels[2], self.channels[3], 3)
        self.stage4 = self._add_stage(self.channels[3], self.channels[4], 3)
        self._init_params()

    def forward(self, x):
        x = self.stage0(x)  # 64, 1/2
        x = self.stage1(x)  # 128, 1/4
        x = self.stage2(x)  # 256, 1/8
        x = self.stage3(x)  # 512, 1/16
        x = self.stage4(x)  # 512, 1/32
        return x

    def _add_stage(self, in_ch, out_ch, repeat_time):
        assert repeat_time > 0 and isinstance(repeat_time, int)
        layers = [
            conv3x3(in_ch, out_ch),
            nn.ReLU(inplace=True),
        ]
        for _ in range(repeat_time - 1):
            layers.extend([
                conv3x3(out_ch, out_ch),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')


class MobileNetV1(nn.Module):

    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.channels = [64, 128, 256, 512, 1024]
        self.stage0 = nn.Sequential(
            conv3x3(3, 32, 2),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            DepthwiseSeparableConv(32, self.channels[0], 1),
        )
        self.stage1 = self._add_stage(self.channels[0], self.channels[1], 2)
        self.stage2 = self._add_stage(self.channels[1], self.channels[2], 2)
        self.stage3 = self._add_stage(self.channels[2], self.channels[3], 6)
        self.stage4 = self._add_stage(self.channels[3], self.channels[4], 2)

        self._init_params()

    def forward(self, x):
        x = self.stage0(x)  # 64, 1/2
        x = self.stage1(x)  # 128, 1/4
        x = self.stage2(x)  # 256, 1/8
        x = self.stage3(x)  # 512, 1/16
        x = self.stage4(x)  # 1024, 1/32
        return x

    def _add_stage(self, in_ch, out_ch, repeat_time):
        assert repeat_time > 0 and isinstance(repeat_time, int)
        layers = [DepthwiseSeparableConv(in_ch, out_ch, 2)]
        for _ in range(repeat_time - 1):
            layers.append(DepthwiseSeparableConv(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResNet50S(nn.Module):

    def __init__(self):
        super(ResNet50S, self).__init__()
        self.channels = [64, 256, 512, 1024, 2048]
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
            # nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = self._add_stage(ResidualBlock, self.channels[0],
                                      self.channels[1], 2, 3)
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

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)