import torch
from torch import nn
import torch.nn.functional as F
from .backbone import ModifiedAlignedXception
from .utils import ASPP


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = ModifiedAlignedXception()
        self.aspp = ASPP(2048)

    def forward(self, x):
        x, low_level_features = self.backbone(x)
        aspp_out = self.aspp(x)
        return low_level_features, aspp_out


class Decoder(nn.Module):

    def __init__(self, num_classes=1):
        super(Decoder, self).__init__()
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(128, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.concat_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.outputs = nn.Conv2d(256, num_classes, 1, bias=False)

    def forward(self, low_level_features, aspp_out):
        low_level_features = self.low_level_conv(low_level_features)
        aspp_out = F.interpolate(aspp_out,
                                 scale_factor=4,
                                 mode='bilinear',
                                 align_corners=False)
        out = torch.cat((aspp_out, low_level_features), dim=1)
        out = self.concat_conv(out)
        out = F.interpolate(out,
                            scale_factor=4,
                            mode='bilinear',
                            align_corners=False)
        return self.outputs(out)


class DeepLabV3Plus(nn.Module):

    def __init__(self, num_classes=1):
        super(DeepLabV3Plus, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        low_level_features, aspp_out = self.encoder(x)
        return self.decoder(low_level_features, aspp_out)
