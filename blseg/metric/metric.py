import torch
import numpy as np


class PixelAccuracy(object):

    def __init__(self, ignore_index=-100, eps=1e-7):
        self.num_correct = 0
        self.num_instance = 0
        self.ignore_index = ignore_index
        self.eps = eps

    def update(self, pred, target):
        ignore_mask = target != self.ignore_index
        if pred.size(1) == 1:
            pred = torch.sigmoid(pred)
            pred = pred > 0.5
        else:
            pred = torch.argmax(pred, dim=1)
        self.num_correct += ((pred.long() == target.long()) *
                             ignore_mask).sum().item()
        self.num_instance += ignore_mask.sum().item()

    def get(self):
        return self.num_correct / (self.num_instance + self.eps)

    def reset(self):
        self.num_correct = 0
        self.num_instance = 0


class MeanIoU(object):

    def __init__(self, num_classes, ignore_background=False, eps=1e-7):
        if num_classes == 1:
            self.num_classes = num_classes + 1
        else:
            self.num_classes = num_classes
        self.num_intersection = np.zeros(self.num_classes)
        self.num_union = np.zeros(self.num_classes)
        self.ignore_background = ignore_background
        self.eps = eps

    def update(self, pred, target):
        if pred.size(1) == 1:
            pred = torch.sigmoid(pred)
            pred = pred > 0.5
        else:
            pred = torch.argmax(pred, dim=1)
        for cur_cls in range(self.num_classes):
            pred_mask = (pred == cur_cls).byte()
            target_mask = (target == cur_cls).byte()

            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()
            self.num_intersection[cur_cls] += intersection.item()
            self.num_union[cur_cls] += union.item()

    def get(self):
        if self.ignore_background:
            return (self.num_intersection[1:] /
                    (self.num_union[1:] + self.eps)).mean()
        else:
            return (self.num_intersection / (self.num_union + self.eps)).mean()

    def reset(self):
        self.num_intersection = np.zeros(self.num_classes)
        self.num_union = np.zeros(self.num_classes)
