import torch
from torch import nn
from backbone import VGG16, ResNet34


class FCN(nn.Module):

    def __init__(self, backbone='vgg16', num_classes=1):
        assert backbone == 'vgg16' or backbone == 'resnet34'
        super(FCN, self).__init__()
        if backbone == 'vgg16':
            self.backbone = VGG16()
        else:
            self.backbone = ResNet34()
        self.backbone.stage0[0].padding = (
            self.backbone.stage0[0].padding[0] + 99,
            self.backbone.stage0[0].padding[1] + 99)
        self.fc = nn.Sequential(
            nn.Conv2d(512, 4096, 7, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.score_fc = nn.Conv2d(4096, num_classes, 1, bias=False)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1, bias=False)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1, bias=False)

        self.score2 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, bias=False)
        self.score4 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, bias=False)
        self.score8 = nn.ConvTranspose2d(
            num_classes, num_classes, 16, stride=8, bias=False)

    def forward(self, x):
        out = self.backbone.stage0(x)
        out = self.backbone.stage1(out)
        pool3_out = self.backbone.stage2(out)
        pool4_out = self.backbone.stage3(pool3_out)
        out = self.backbone.stage4(pool4_out)
        out = self.fc(out)
        out = self.score_fc(out)
        score2 = self.score2(out)
        print(score2.shape)

        score_pool4 = self.score_pool4(pool4_out)
        print(score_pool4.shape)
        score_pool4 = score_pool4[:, :, 5:5 + score2.size(2), 5:5 +
                                  score2.size(3)]
        print(score_pool4.shape)
        fuse1 = score2 + score_pool4
        score4 = self.score4(fuse1)
        print(score4.shape)

        score_pool3 = self.score_pool3(pool3_out)
        print(score_pool3.shape)
        score_pool3 = score_pool3[:, :, 9:9 + score4.size(2), 9:9 +
                                  score4.size(3)]
        print(score_pool3.shape)
        fuse2 = score4 + score_pool3
        score8 = self.score8(fuse2)
        print(score8.shape)
        x = score8[:, :, 31:31 + x.size(2), 31:31 + x.size(3)]
        print(x.shape)
        return x
