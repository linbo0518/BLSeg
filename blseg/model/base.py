import torch
from torch import nn
from ..backbone import *


class SegBaseModule(nn.Module):

    def __init__(self, num_classes=1):
        super(SegBaseModule, self).__init__()
        self.num_classes = num_classes

    def _get_backbone(self, backbone_name):
        if backbone_name == 'vgg16':
            return VGG16()
        elif backbone_name == 'mobilenetv1':
            return MobileNetV1()
        elif backbone_name == 'resnet50':
            return ResNet50S()
        elif backbone_name == 'mobilenetv2':
            return MobileNetV2()
        elif backbone_name == 'xception':
            return ModifiedAlignedXception()
        else:
            raise NotImplementedError()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)