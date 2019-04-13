import torch
from torch import nn
from .utils import conv3x3, DepthwiseSeparableConv
from .base import BackboneBaseModule


class XceptionBlock(nn.Module):

    def __init__(self,
                 in_ch,
                 out_chs,
                 stride,
                 residual_type,
                 first_relu=True,
                 dilation=1):
        assert isinstance(out_chs, list)
        assert residual_type == 'conv' or residual_type == 'sum' or residual_type == 'none'
        super(XceptionBlock, self).__init__()

        xception_block = []
        if first_relu:
            xception_block.append(nn.ReLU())
        xception_block.extend([
            DepthwiseSeparableConv(in_ch,
                                   out_chs[0],
                                   stride=1,
                                   relu6=False,
                                   last_relu=False),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_chs[0],
                                   out_chs[1],
                                   stride=1,
                                   relu6=False,
                                   last_relu=False),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_chs[1],
                                   out_chs[2],
                                   stride=stride,
                                   dilation=dilation,
                                   relu6=False,
                                   last_relu=False)
        ])
        if residual_type == 'none':
            xception_block.append(nn.ReLU(inplace=True))
        self.xception_block = nn.Sequential(*xception_block)

        self.residual_type = residual_type
        if self.residual_type == 'conv':
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_chs[2], 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chs[2]),
            )

    def forward(self, x):
        residual = x

        x = self.xception_block(x)

        if self.residual_type == 'conv':
            residual = self.residual(residual)
            x += residual
        elif self.residual_type == 'sum':
            x += residual

        return x


class ModifiedAlignedXception(BackboneBaseModule):

    def __init__(self):
        super(ModifiedAlignedXception, self).__init__()
        self.channels = [64, 128, 256, 728, 2048]
        self.strides = [2, 4, 8, 16, 32]
        self.stage0 = nn.Sequential(
            conv3x3(3, 32, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv3x3(32, self.channels[0]),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = XceptionBlock(
            self.channels[0],
            [self.channels[1], self.channels[1], self.channels[1]],
            stride=2,
            residual_type='conv',
            first_relu=False)
        self.stage2 = XceptionBlock(
            self.channels[1],
            [self.channels[2], self.channels[2], self.channels[2]],
            stride=2,
            residual_type='conv',
            first_relu=True)

        layers = [
            XceptionBlock(
                self.channels[2],
                [self.channels[3], self.channels[3], self.channels[3]],
                stride=2,
                residual_type='conv',
                first_relu=True)
        ]
        for _ in range(16):
            layers.append(
                XceptionBlock(
                    self.channels[3],
                    [self.channels[3], self.channels[3], self.channels[3]],
                    stride=1,
                    residual_type='sum',
                    first_relu=True))
        self.stage3 = nn.Sequential(*layers)
        self.stage4 = nn.Sequential(
            XceptionBlock(self.channels[3], [
                self.channels[3], self.channels[4] - 1024,
                self.channels[4] - 1024
            ],
                          stride=2,
                          residual_type='conv',
                          first_relu=True),
            XceptionBlock(self.channels[4] - 1024, [
                self.channels[4] - 512, self.channels[4] - 512, self.channels[4]
            ],
                          stride=1,
                          residual_type='none',
                          first_relu=False),
        )

        self._init_params()

    def forward(self, x):
        x = self.stage0(x)  # 64, 1/2
        x = self.stage1(x)  # 128, 1/4
        x = self.stage2(x)  # 256, 1/8
        x = self.stage3(x)  # 728, 1/16
        x = self.stage4(x)  # 2048, 1/32
        return x

    def _change_downsample(self, params):
        self.stage3[0].xception_block[5].dwconv.stride = (params[0], params[0])
        self.stage3[0].residual[0].stride = (params[0], params[0])
        self.stage4[0].xception_block[5].dwconv.stride = (params[1], params[1])
        self.stage4[0].residual[0].stride = (params[1], params[1])
