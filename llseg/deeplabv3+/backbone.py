import torch
from torch import nn
from .utils import XceptionBlock


class ModifiedAlignedXception(nn.Module):

    def __init__(self):
        super(ModifiedAlignedXception, self).__init__()
        self.entry_flow1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            XceptionBlock(64, [128, 128, 128],
                          stride=2,
                          residual_type='conv',
                          first_relu=False),
        )
        self.entry_flow2 = nn.Sequential(
            XceptionBlock(128, [256, 256, 256],
                          stride=2,
                          residual_type='conv',
                          first_relu=True),
            XceptionBlock(256, [728, 728, 728],
                          stride=2,
                          residual_type='conv',
                          first_relu=True),
        )
        middle_flow = []
        for _ in range(16):
            middle_flow.append(
                XceptionBlock(728, [728, 728, 728],
                              stride=1,
                              residual_type='sum',
                              first_relu=True))
        self.middle_flow = nn.Sequential(*middle_flow,)
        self.exit_flow = nn.Sequential(
            XceptionBlock(728, [728, 1024, 1024],
                          stride=1,
                          residual_type='conv',
                          first_relu=True),
            XceptionBlock(1024, [1536, 1536, 2048],
                          stride=1,
                          residual_type='none',
                          first_relu=False,
                          dilation=2),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.entry_flow1(x)
        low_level_featuers = x
        x = self.entry_flow2(x)
        x = self.middle_flow(x)
        return self.exit_flow(x), low_level_featuers
