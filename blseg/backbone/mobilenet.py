import torch
from torch import nn
from .utils import conv3x3, DepthwiseSeparableConv


class MobileNetV1(nn.Module):

    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.channels = [64, 128, 256, 512, 1024]
        self.strides = [2, 4, 8, 16, 32]
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

    def change_output_stride(self, output_stride):
        assert output_stride in [16, 32]
        if output_stride == 16:
            self.stage4[0].dwconv.stride = (1, 1)
            self.stage4[0].dwconv.padding = (2, 2)
            self.stage4[0].dwconv.dilation = (2, 2)
            self.stage4[1].dwconv.padding = (2, 2)
            self.stage4[1].dwconv.dilation = (2, 2)
        elif output_stride == 32:
            self.stage4[0].dwconv.stride = (2, 2)
            self.stage4[0].dwconv.padding = (1, 1)
            self.stage4[0].dwconv.dilation = (1, 1)
            self.stage4[1].dwconv.padding = (1, 1)
            self.stage4[1].dwconv.dilation = (1, 1)