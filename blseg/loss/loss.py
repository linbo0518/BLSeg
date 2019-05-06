import torch
from torch import nn
from ..utils import one_hot


def _ohem_mask(loss, ohem_ratio):
    with torch.no_grad():
        values, _ = torch.topk(loss.reshape(-1),
                               int(loss.nelement() * ohem_ratio))
        mask = loss >= values[-1]
    return mask.float()


class BCEWithLogitsLossWithOHEM(nn.Module):

    def __init__(self, ohem_ratio=1.0, pos_weight=None, eps=1e-7):
        super(BCEWithLogitsLossWithOHEM, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none',
                                              pos_weight=pos_weight)
        self.ohem_ratio = ohem_ratio
        self.eps = eps

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        mask = _ohem_mask(loss, self.ohem_ratio)
        loss = loss * mask
        return loss.sum() / (mask.sum() + self.eps)


class CrossEntropyLossWithOHEM(nn.Module):

    def __init__(self, ohem_ratio=1.0, ignore_index=-100, eps=1e-7):
        super(CrossEntropyLossWithOHEM, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                             reduction='none')
        self.ohem_ratio = ohem_ratio
        self.eps = eps

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        mask = _ohem_mask(loss, self.ohem_ratio)
        loss = loss * mask
        return loss.sum() / (mask.sum() + self.eps)


class DiceLoss(nn.Module):

    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = torch.cat((1 - pred, pred), dim=1)
        target = one_hot(target, num_classes=2)
        intersection = (pred * target).sum((0, 2, 3))
        loss = 1 - ((2. * intersection) / (pred.sum((0, 2, 3)) + target.sum(
            (0, 2, 3)) + self.eps)).mean()
        return loss