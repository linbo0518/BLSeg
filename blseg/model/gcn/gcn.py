import torch
from torch import nn
from ..base import SegBaseModule
from .utils import GCNModule, BRModule, Pipeline


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
