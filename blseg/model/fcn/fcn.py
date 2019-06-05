import torch
from torch import nn
from ..base import SegBaseModule


class FCN(SegBaseModule):

    def __init__(self,
                 backbone='vgg16',
                 num_classes=21,
                 dilations=[1, 1, 1, 1, 1]):
        assert backbone in [
            'vgg16', 'resnet34', 'resnet50', 'se_resnet34', 'se_resnet50',
            'mobilenet_v1', 'mobilenet_v2', 'xception'
        ]
        super(FCN, self).__init__(num_classes)
        self.backbone = self._get_backbone(backbone)
        self.backbone.change_dilation(dilations)
        self.backbone.stage0[0].padding = (self.backbone.stage0[0].padding[0] +
                                           99,
                                           self.backbone.stage0[0].padding[1] +
                                           99)
        self.fc = nn.Sequential(
            nn.Conv2d(self.backbone.channels[4], 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.score_fc = nn.Conv2d(4096, num_classes, 1)
        self.score_pool4 = nn.Conv2d(self.backbone.channels[3], num_classes, 1)

        self.score2 = nn.ConvTranspose2d(num_classes,
                                         num_classes,
                                         4,
                                         stride=2,
                                         bias=False)
        self.score16 = nn.ConvTranspose2d(num_classes,
                                          num_classes,
                                          32,
                                          stride=16,
                                          bias=False)

        self._init_params()

    def forward(self, x):
        out = self.backbone.stage0(x)
        out = self.backbone.stage1(out)
        out = self.backbone.stage2(out)
        pool4_out = self.backbone.stage3(out)
        out = self.backbone.stage4(pool4_out)
        out = self.fc(out)
        out = self.score_fc(out)
        score2 = self.score2(out)

        score_pool4 = self.score_pool4(pool4_out)
        score_pool4 = score_pool4[:, :, 5:5 + score2.size(2), 5:5 +
                                  score2.size(3)]

        score16 = self.score16(score2 + score_pool4)
        x = score16[:, :, 27:27 + x.size(2), 27:27 + x.size(3)]
        return x

    def reset_classes(self, num_classes):
        self.num_classes = num_classes
        self.score_fc = nn.Conv2d(4096, num_classes, 1)
        self.score_pool4 = nn.Conv2d(self.backbone.channels[3], num_classes, 1)

        self.score2 = nn.ConvTranspose2d(num_classes,
                                         num_classes,
                                         4,
                                         stride=2,
                                         bias=False)
        self.score16 = nn.ConvTranspose2d(num_classes,
                                          num_classes,
                                          32,
                                          stride=16,
                                          bias=False)