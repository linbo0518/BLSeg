import torch
from torch import nn
from utils import ResidualBlock


class ResNet34(nn.Module):

    def __init__(self):
        super(ResNet34, self).__init__()
        self.channel = 64
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, stride=2, padding=1),
            # remove maxpool to keep featuremap size not too small
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
