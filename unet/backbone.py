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
        )
        self.stage1 = self._add_stage(ResidualBlock, 64, 3)
        self.stage2 = self._add_stage(ResidualBlock, 128, 4)
        self.stage3 = self._add_stage(ResidualBlock, 256, 6)
        self.stage4 = self._add_stage(ResidualBlock, 512, 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Linear(512, 1000)

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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

    def _add_stage(self, block, channel, repeat):
        block_list = []
        stride = 1
        for _ in range(repeat):
            if self.channel != channel:
                stride = 2
            else:
                stride = 1
            block_list.append(block(self.channel, channel, stride))
            self.channel = channel
        return nn.Sequential(*block_list)
