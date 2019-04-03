import torch
from torch import nn
from utils import conv3x3, ResidualBlock


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.stage0 = self._add_stage(3, 64, 2)
        self.stage1 = self._add_stage(64, 128, 2)
        self.stage2 = self._add_stage(128, 256, 3)
        self.stage3 = self._add_stage(256, 512, 3)
        self.stage4 = self._add_stage(512, 512, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
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


class ResNet34(nn.Module):

    def __init__(self):
        super(ResNet34, self).__init__()
        self.channel = 64
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = self._add_stage(ResidualBlock, 64, 64, 1, 3)
        self.stage2 = self._add_stage(ResidualBlock, 64, 128, 2, 4)
        self.stage3 = self._add_stage(ResidualBlock, 128, 256, 2, 6)
        self.stage4 = self._add_stage(ResidualBlock, 256, 512, 2, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

    def _add_stage(self, block, in_ch, out_ch, stride, repeat_times):
        assert repeat_times > 0 and isinstance(repeat_times, int)
        layers = [block(in_ch, out_ch, stride)]
        for _ in range(repeat_times - 1):
            layers.append(block(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
