import torch
from torch import nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_separable_conv2d = nn.Sequential(
            nn.Conv2d(in_ch,
                      in_ch,
                      3,
                      stride=stride,
                      padding=dilation,
                      dilation=dilation,
                      groups=in_ch,
                      bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.depthwise_separable_conv2d(x)


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
            DepthwiseSeparableConv(in_ch, out_chs[0], stride=1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_chs[0], out_chs[1], stride=1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_chs[1],
                                   out_chs[2],
                                   stride=stride,
                                   dilation=dilation)
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


class ASPP(nn.Module):

    def __init__(self, in_ch):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_ch, 256, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.aspp_out = nn.Sequential(
            nn.Conv2d(1280, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = (x.size(2), x.size(3))
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        aspp5 = self.aspp5(x)
        aspp5 = F.interpolate(aspp5,
                              size=size,
                              mode='bilinear',
                              align_corners=False)
        all_aspp = torch.cat((aspp1, aspp2, aspp3, aspp4, aspp5), dim=1)
        return self.aspp_out(all_aspp)
