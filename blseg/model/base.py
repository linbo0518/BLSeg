import torch
from torch import nn
from ..backbone import *


class SegBaseModule(nn.Module):
    def __init__(self, num_classes=1):
        super(SegBaseModule, self).__init__()
        self.num_classes = num_classes

    def train_backbone(self):
        self.backbone.train()
        self.backbone.requires_grad_(True)

    def freeze_backbone(self):
        self.backbone.eval()
        self.backbone.requires_grad_(False)

    def train_batch_norm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                m.requires_grad_(True)

    def freeze_batch_norm(self,
                          freeze_running_mean_var=True,
                          freeze_gamma_beta=True):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if freeze_running_mean_var:
                    m.eval()
                if freeze_gamma_beta:
                    m.requires_grad_(False)

    @torch.no_grad()
    def precise_batch_norm(self, dataloader):
        device = next(self.parameters()).device
        bns = [m for m in self.modules() if isinstance(m, nn.BatchNorm2d)]

        if len(bns) == 0:
            return

        running_means = [torch.zeros_like(bn.running_mean) for bn in bns]
        running_vars = [torch.zeros_like(bn.running_var) for bn in bns]
        momentums = [bn.momentum for bn in bns]

        for bn in bns:
            bn.momentum = 1.0

        for batch_idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            self.forward(inputs)
            for bn_idx, bn in enumerate(bns):
                running_means[bn_idx] += (
                    bn.running_mean - running_means[bn_idx]) / (batch_idx + 1)
                running_vars[bn_idx] += (
                    bn.running_var - running_vars[bn_idx]) / (batch_idx + 1)

        for bn_idx, bn in bns:
            bn.running_mean = running_means[bn_idx]
            bn.running_var = running_vars[bn_idx]
            bn.momentum = momentums[bn_idx]

    def load_parameters(self, filename, map_location=None, strict=True):
        self.load_state_dict(torch.load(filename, map_location=map_location),
                             strict=strict)

    def load_backbone_parameters(self,
                                 filename,
                                 map_location=None,
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

    def _init_params(self, method="msra", zero_gamma=True):
        method = method.lower()
        assert method in ("xavier", "msra")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if method == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                if method == "msra":
                    nn.init.kaiming_normal_(m.weight,
                                            mode='fan_out',
                                            nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.BatchNorm2d):
                last_bn = zero_gamma and hasattr(m, "last_bn")
                nn.init.constant_(m.weight, 0.0 if last_bn else 1.0)
                nn.init.constant_(m.bias, 0.0)