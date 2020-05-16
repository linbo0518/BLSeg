import torch
from torch import nn
from .utils import conv3x3, DepthwiseSeparableConv
from .base import BackboneBaseModule

__all__ = [
    "MobileNetV1",
    "MobileNetV2",
]


class LinearBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, t, stride):
        super(LinearBottleneck, self).__init__()
        self.do_residual = in_ch == out_ch and stride == 1
        self.conv1 = nn.Conv2d(in_ch, in_ch * t, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch * t)
        self.relu = nn.ReLU6(inplace=True)
        self.dsconv = DepthwiseSeparableConv(in_ch * t,
                                             out_ch,
                                             stride=stride,
                                             relu6=True,
                                             last_relu=False,
                                             last_bn=self.do_residual)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dsconv(x)
        if self.do_residual:
            x += residual
        return x


class MobileNetV1(BackboneBaseModule):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.channels = [64, 128, 256, 512, 1024]
        self.strides = [2, 4, 8, 16, 32]
        self.stage0 = nn.Sequential(
            conv3x3(3, 32, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, self.channels[0], 1, relu6=False),
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
        layers = [DepthwiseSeparableConv(in_ch, out_ch, 2, relu6=False)]
        for _ in range(repeat_time - 1):
            layers.append(DepthwiseSeparableConv(out_ch, out_ch, relu6=False))
        return nn.Sequential(*layers)

    def _change_downsample(self, params):
        self.stage3[0].dwconv.stride = (params[0], params[0])
        self.stage4[0].dwconv.stride = (params[1], params[1])


class MobileNetV2(BackboneBaseModule):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.channels = [16, 24, 32, 96, 1280]
        self.strides = [2, 4, 8, 16, 32]
        self.stage0 = nn.Sequential(
            conv3x3(3, 32, 2),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            LinearBottleneck(32, self.channels[0], 1, 1),
        )
        self.stage1 = self._add_stage(LinearBottleneck, self.channels[0],
                                      self.channels[1], 6, 2, 2)
        self.stage2 = self._add_stage(LinearBottleneck, self.channels[1],
                                      self.channels[2], 6, 2, 3)
        self.stage3 = self._add_stage(LinearBottleneck, self.channels[2],
                                      self.channels[2] * 2, 6, 2, 4)
        for layer in self._add_stage(LinearBottleneck, self.channels[2] * 2,
                                     self.channels[3], 6, 1, 3):
            self.stage3.add_module(str(len(self.stage3)), layer)

        self.stage4 = self._add_stage(LinearBottleneck, self.channels[3],
                                      int(self.channels[4] / 8), 6, 2, 3)
        for layer in self._add_stage(LinearBottleneck,
                                     int(self.channels[4] / 8),
                                     int(self.channels[4] / 4), 6, 1, 1):
            self.stage4.add_module(str(len(self.stage4)), layer)
        self.stage4.add_module(
            str(len(self.stage4)),
            nn.Sequential(
                nn.Conv2d(int(self.channels[4] / 4),
                          self.channels[4],
                          1,
                          bias=False),
                nn.BatchNorm2d(self.channels[4]),
                nn.ReLU6(inplace=True),
            ))

        self._init_params()

    def forward(self, x):
        x = self.stage0(x)  # 16, 1/2
        x = self.stage1(x)  # 24, 1/4
        x = self.stage2(x)  # 32, 1/8
        x = self.stage3(x)  # 96, 1/16
        x = self.stage4(x)  # 1280, 1/32
        return x

    def _add_stage(self, block, in_ch, out_ch, t, stride, repeat_time):
        assert repeat_time > 0 and isinstance(repeat_time, int)
        layers = [block(in_ch, out_ch, t, stride)]
        for _ in range(repeat_time - 1):
            layers.append(block(out_ch, out_ch, t, 1))
        return nn.Sequential(*layers)

    def _change_downsample(self, params):
        self.stage3[0].dsconv.dwconv.stride = (params[0], params[0])
        self.stage4[0].dsconv.dwconv.stride = (params[1], params[1])
