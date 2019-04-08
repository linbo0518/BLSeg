import torch
from torch import nn
from .backbone import MobileNetV1, VGG16, ResNet34


class FCN(nn.Module):

    def __init__(self, backbone='vgg16', num_classes=1):
        assert backbone == 'vgg16' or backbone == 'resnet34' or backbone == 'mobilenetv1'
        super(FCN, self).__init__()
        if backbone == 'vgg16':
            self.backbone = VGG16()
            self.fc_in_ch = 512
            self.score_pool4_in_ch = 512
            self.score_pool3_in_ch = 256
        elif backbone == 'resnet34':
            self.backbone = ResNet34()
            self.fc_in_ch = 512
            self.score_pool4_in_ch = 256
            self.score_pool3_in_ch = 128
        elif backbone == 'mobilenetv1':
            self.backbone = MobileNetV1()
            self.fc_in_ch = 1024
            self.score_pool4_in_ch = 512
            self.score_pool3_in_ch = 256

        self.backbone.stage0[0].padding = (self.backbone.stage0[0].padding[0] +
                                           99,
                                           self.backbone.stage0[0].padding[1] +
                                           99)
        self.fc = nn.Sequential(
            nn.Conv2d(self.fc_in_ch, 4096, 7, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.score_fc = nn.Conv2d(4096, num_classes, 1, bias=False)
        self.score_pool4 = nn.Conv2d(self.score_pool4_in_ch,
                                     num_classes,
                                     1,
                                     bias=False)
        self.score_pool3 = nn.Conv2d(self.score_pool3_in_ch,
                                     num_classes,
                                     1,
                                     bias=False)

        self.score2 = nn.ConvTranspose2d(num_classes,
                                         num_classes,
                                         4,
                                         stride=2,
                                         bias=False)
        self.score4 = nn.ConvTranspose2d(num_classes,
                                         num_classes,
                                         4,
                                         stride=2,
                                         bias=False)
        self.score8 = nn.ConvTranspose2d(num_classes,
                                         num_classes,
                                         16,
                                         stride=8,
                                         bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.backbone.stage0(x)
        out = self.backbone.stage1(out)
        pool3_out = self.backbone.stage2(out)
        pool4_out = self.backbone.stage3(pool3_out)
        out = self.backbone.stage4(pool4_out)
        out = self.fc(out)
        out = self.score_fc(out)
        score2 = self.score2(out)

        score_pool4 = self.score_pool4(pool4_out)
        score_pool4 = score_pool4[:, :, 5:5 + score2.size(2), 5:5 +
                                  score2.size(3)]
        fuse1 = score2 + score_pool4
        score4 = self.score4(fuse1)

        score_pool3 = self.score_pool3(pool3_out)
        score_pool3 = score_pool3[:, :, 9:9 + score4.size(2), 9:9 +
                                  score4.size(3)]
        fuse2 = score4 + score_pool3
        score8 = self.score8(fuse2)

        x = score8[:, :, 31:31 + x.size(2), 31:31 + x.size(3)]
        return x
