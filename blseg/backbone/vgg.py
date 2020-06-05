import torch
from torch import nn
from .base import BackboneBaseModule

__all__ = ["VGG16"]


def _add_stage(in_ch, out_ch, repeat_time):
    assert repeat_time > 0 and isinstance(repeat_time, int)
    layers = [
        nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        nn.ReLU(inplace=True),
    ]
    for _ in range(repeat_time - 1):
        layers.extend([
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ])
    layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
    return nn.Sequential(*layers)


class VGG16(BackboneBaseModule):
    def __init__(self):
        super(VGG16, self).__init__()
        self.channels = [64, 128, 256, 512, 512]
        self.strides = [2, 4, 8, 16, 32]
        self.stage0 = _add_stage(3, self.channels[0], 2)
        self.stage1 = _add_stage(self.channels[0], self.channels[1], 2)
        self.stage2 = _add_stage(self.channels[1], self.channels[2], 3)
        self.stage3 = _add_stage(self.channels[2], self.channels[3], 3)
        self.stage4 = _add_stage(self.channels[3], self.channels[4], 3)

        self._init_params()

    def forward(self, x):
        x = self.stage0(x)  # 64, 1/2
        x = self.stage1(x)  # 128, 1/4
        x = self.stage2(x)  # 256, 1/8
        x = self.stage3(x)  # 512, 1/16
        x = self.stage4(x)  # 512, 1/32
        return x

    def _change_downsample(self, params):
        self.stage3[6].kernel_size = params[0]
        self.stage3[6].stride = params[0]
        self.stage4[6].kernel_size = params[1]
        self.stage4[6].stride = params[1]


class VGG19(BackboneBaseModule):
    def __init__(self):
        super(VGG19, self).__init__()
        self.channels = [64, 128, 256, 512, 512]
        self.strides = [2, 4, 8, 16, 32]
        self.stage0 = _add_stage(3, self.channels[0], 2)
        self.stage1 = _add_stage(self.channels[0], self.channels[1], 2)
        self.stage2 = _add_stage(self.channels[1], self.channels[2], 4)
        self.stage3 = _add_stage(self.channels[2], self.channels[3], 4)
        self.stage4 = _add_stage(self.channels[3], self.channels[4], 4)

        self._init_params()

    def forward(self, x):
        x = self.stage0(x)  # 64, 1/2
        x = self.stage1(x)  # 128, 1/4
        x = self.stage2(x)  # 256, 1/8
        x = self.stage3(x)  # 512, 1/16
        x = self.stage4(x)  # 512, 1/32
        return x

    def _change_downsample(self, params):
        self.stage3[8].kernel_size = params[0]
        self.stage3[8].stride = params[0]
        self.stage4[8].kernel_size = params[1]
        self.stage4[8].stride = params[1]