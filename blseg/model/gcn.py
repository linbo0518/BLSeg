import torch
from torch import nn
from ..backbone.utils import conv3x3
from .base import SegBaseModule

__all__ = ["GCN"]


class GCNModule(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super(GCNModule, self).__init__()
        padding = (k - 1) // 2
        self.gcn1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (k, 1), padding=(padding, 0), bias=False),
            nn.Conv2d(out_ch, out_ch, (1, k), padding=(0, padding),
                      bias=False),
        )
        self.gcn2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, k), padding=(0, padding), bias=False),
            nn.Conv2d(out_ch, out_ch, (k, 1), padding=(padding, 0),
                      bias=False),
        )

    def forward(self, x):
        gcn1 = self.gcn1(x)
        gcn2 = self.gcn2(x)
        return gcn1 + gcn2


class BRModule(nn.Module):
    def __init__(self, in_ch):
        super(BRModule, self).__init__()
        self.br = nn.Sequential(
            conv3x3(in_ch, in_ch),
            nn.ReLU(inplace=True),
            conv3x3(in_ch, in_ch),
        )

    def forward(self, x):
        br = self.br(x)
        return x + br


class Pipeline(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super(Pipeline, self).__init__()
        self.gcn = GCNModule(in_ch, out_ch, k)
        self.br1 = BRModule(out_ch)
        self.br2 = BRModule(out_ch)
        self.deconv = nn.ConvTranspose2d(out_ch, out_ch, 2, 2, bias=False)

    def forward(self, x, fuse):
        x = self.gcn(x)
        x = self.br1(x)
        x = self.br2(x + fuse)
        x = self.deconv(x)
        return x


class GCN(SegBaseModule):
    def __init__(self,
                 backbone='resnet50',
                 num_classes=21,
                 dilations=(1, 1, 1, 1, 1),
                 k=9):
        assert backbone in [
            'vgg16', 'resnet34', 'resnet50', 'se_resnet34', 'se_resnet50',
            'mobilenet_v1', 'mobilenet_v2', 'xception'
        ]
        super(GCN, self).__init__(num_classes)
        self.k = k
        self.backbone = self._get_backbone(backbone)
        self.backbone.change_dilation(dilations)
        self.pipeline1 = Pipeline(self.backbone.channels[1],
                                  num_classes,
                                  k=self.k)
        self.pipeline2 = Pipeline(self.backbone.channels[2],
                                  num_classes,
                                  k=self.k)
        self.pipeline3 = Pipeline(self.backbone.channels[3],
                                  num_classes,
                                  k=self.k)
        self.pipeline4 = nn.Sequential(
            GCNModule(self.backbone.channels[4], num_classes, k=self.k),
            BRModule(num_classes),
            nn.ConvTranspose2d(num_classes, num_classes, 2, 2, bias=False),
        )
        self.out = nn.Sequential(
            BRModule(num_classes),
            nn.ConvTranspose2d(num_classes, num_classes, 2, 2, bias=False),
            BRModule(num_classes))

        self._init_params()

    def forward(self, x):
        x = self.backbone.stage0(x)
        stage1 = self.backbone.stage1(x)
        stage2 = self.backbone.stage2(stage1)
        stage3 = self.backbone.stage3(stage2)
        stage4 = self.backbone.stage4(stage3)
        pipeline4 = self.pipeline4(stage4)
        pipeline3 = self.pipeline3(stage3, pipeline4)
        pipeline2 = self.pipeline2(stage2, pipeline3)
        pipeline1 = self.pipeline1(stage1, pipeline2)
        out = self.out(pipeline1)
        return out

    def reset_classes(self, num_classes):
        self.num_classes = num_classes
        self.pipeline1 = Pipeline(self.backbone.channels[1],
                                  num_classes,
                                  k=self.k)
        self.pipeline2 = Pipeline(self.backbone.channels[2],
                                  num_classes,
                                  k=self.k)
        self.pipeline3 = Pipeline(self.backbone.channels[3],
                                  num_classes,
                                  k=self.k)
        self.pipeline4 = nn.Sequential(
            GCNModule(self.backbone.channels[4], num_classes, k=self.k),
            BRModule(num_classes),
            nn.ConvTranspose2d(num_classes, num_classes, 2, 2, bias=False),
        )
        self.out = nn.Sequential(
            BRModule(num_classes),
            nn.ConvTranspose2d(num_classes, num_classes, 2, 2, bias=False),
            BRModule(num_classes))
