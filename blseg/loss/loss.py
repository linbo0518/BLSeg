import torch
from torch import nn
import torch.nn.functional as F


def _ohem_mask(loss, ohem_ratio):
    with torch.no_grad():
        values, _ = torch.topk(loss.reshape(-1),
                               int(loss.nelement() * ohem_ratio))
        mask = loss >= values[-1]
    return mask.float()


class BCEWithLogitsLossWithOHEM(nn.Module):

    def __init__(self, ohem_ratio=1.0, pos_weight=None):
        super(BCEWithLogitsLossWithOHEM, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none',
                                              pos_weight=pos_weight)
        self.ohem_ratio = ohem_ratio

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        if self.ohem_ratio < 1.0:
            mask = _ohem_mask(loss, self.ohem_ratio)
            loss = loss * mask
        return loss.mean()


class CrossEntropyLossWithOHEM(nn.Module):

    def __init__(self, ohem_ratio=1.0, ignore_index=-100):
        super(CrossEntropyLossWithOHEM, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                             reduction='none')
        self.ohem_ratio = ohem_ratio

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        if self.ohem_ratio < 1.0:
            mask = _ohem_mask(loss, self.ohem_ratio)
            loss = loss * mask
        return loss.mean()


class BinaryDiceLossWithOHEM(nn.Module):
    # TODO: waiting for test
    def __init__(self, ohem_ratio=1.0):
        super(BinaryDiceLossWithOHEM, self).__init__()
        self.ohem_ratio = ohem_ratio

    def forward(self, pred, target):
        pass

    def _dice_loss(self, pred, label):
        eps = 1e-12
        pred = torch.sigmoid(pred)
        intersection = (pred * label).sum((1, 2, 3))
        dice_coef = 1 - (2. * intersection) / (pred.sum((1, 2, 3)) + label.sum(
            (1, 2, 3)) + eps)
        return dice_coef


class MultiClassDiceLossWithOHEM(nn.Module):
    # TODO: waiting for test
    def __init__(self, ohem_ratio=1.0, ignore_index=-100):
        super(MultiClassDiceLossWithOHEM, self).__init__()
        self.ohem_ratio = ohem_ratio
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pass