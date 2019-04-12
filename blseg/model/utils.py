from ..backbone import *


def _get_backbone(backbone_name):
    if backbone_name == 'vgg16':
        return VGG16()
    elif backbone_name == 'mobilenetv1':
        return MobileNetV1()
    elif backbone_name == 'resnet50':
        return ResNet50S()
    elif backbone_name == 'xception':
        return ModifiedAlignedXception()
    else:
        raise NotImplementedError()