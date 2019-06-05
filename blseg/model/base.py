import torch
from torch import nn
from ..backbone import *


class SegBaseModule(nn.Module):

    def __init__(self, num_classes=1):
        super(SegBaseModule, self).__init__()
        self.num_classes = num_classes

    def train_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_BN(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def load_parameters(self, filename, map_location=None, strict=True):
        self.load_state_dict(torch.load(filename, map_location=map_location),
                             strict=strict)

    def load_backbone_parameters(self, filename, map_location=None,
                                 strict=True):
        self.backbone.load_parameters(filename, map_location, strict)

    def reset_classes(self, num_classes):
        '''
        Should be overridden by all subclasses
        '''
        self.num_classes = num_classes
        raise NotImplementedError

    def _get_backbone(self, backbone_name):
        if backbone_name == 'vgg16':
            return VGG16()
        elif backbone_name == 'mobilenet_v1':
            return MobileNetV1()
        elif backbone_name == 'resnet34':
            return ResNet34()
        elif backbone_name == 'se_resnet34':
            return ResNet34(use_se=True)
        elif backbone_name == 'resnet50':
            return ResNet50S()
        elif backbone_name == 'se_resnet50':
            return ResNet50S(use_se=True)
        elif backbone_name == 'mobilenet_v2':
            return MobileNetV2()
        elif backbone_name == 'xception':
            return ModifiedAlignedXception()
        else:
            raise NotImplementedError

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