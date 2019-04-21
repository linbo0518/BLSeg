import torch
import numpy as np


class PixelAccuracy(object):

    def __init__(self, ignore_index=-1, eps=1e-12):
        self.num_correct = 0
        self.num_instance = 0
        self.ignore_index = ignore_index
        self.eps = eps

    def update(self, pred, target):
        valid_mask = target != self.ignore_index
        pred = torch.argmax(pred, dim=1)
        self.num_correct += ((pred == target) * valid_mask).sum().item()
        self.num_instance += valid_mask.sum().item()

    def get(self):
        return self.num_correct / (self.num_instance + self.eps)

    def reset(self):
        self.num_correct = 0
        self.num_instance = 0


class MeanIoU(object):

    def __init__(self, num_classes, eps=1e-12):
        self.num_classes = num_classes
        self.num_intersection = np.zeros(self.num_classes)
        self.num_union = np.zeros(self.num_classes)
        self.eps = eps

    def update(self, pred, target):
        pred = torch.argmax(pred, dim=1)
        for cur_cls in range(self.num_classes):
            pred_mask = (pred == cur_cls).byte()
            target_mask = (target == cur_cls).byte()

            intersection = (pred_mask & target_mask).float().sum((0, 1, 2))
            union = (pred_mask | target_mask).float().sum((0, 1, 2))
            self.num_intersection[cur_cls] += intersection.item()
            self.num_union[cur_cls] += union.item()

    def get(self):
        return (self.num_intersection[1:] /
                (self.num_union[1:] + self.eps)).mean()

    def reset(self):
        self.num_intersection = np.zeros(self.num_classes)
        self.num_union = np.zeros(self.num_classes)
