# Docs

* Note: The order of the color channels is **BGR**, please use `cv2.imread` to read image.
* Note: the **mean** is `(0.485, 0.456, 0.406)` and the **std** is `(0.229, 0.224, 0.225)`

Import BLSeg Package

```Python
from blseg import nn
```

## Initialize

Create Model

```Python
num_classes = 21

# Available backbone:
# vgg16, resnet34, resnet50, se_resnet34, se_resnet50,
# mobilenet_v1, mobilenet_v2, xception
net = nn.FCN('vgg16', num_classes)
net = nn.ModernUNet('mobilenet_v1', num_classes)
net = nn.PSPNet('resnet50', num_classes)
net = nn.DeepLabV3Plus('xception', num_classes)
```

Create Loss

```Python
ohem_ratio = 0.7

# Binary Cross Entropy Loss with OHEM
criterion = nn.BCEWithLogitsLossWithOHEM(ohem_ratio)

# Cross Entropy Loss with OHEM
criterion = nn.CrossEntropyLossWithOHEM(ohem_ratio)

# Dice Loss
criterion = nn.DiceLoss()

# Soft-label Cross Entropy Loss with OHEM
criterion = nn.SoftCrossEntropyLossWithOHEM(ohem_ratio)
```

Create Metric

```Python
# Loss Meter
loss_meter = nn.LossMeter()

# Pixel Accuracy
pixacc = nn.PixelAccuracy()

# Mean IoU
miou = nn.MeanIoU(num_classes)
```

## Usage

Model API

```Python
# train backbone (default is train the entire net)
net.train_backbone()

# freeze backbone, only train the segmentation head
net.freeze_backbone()

# train BatchNorm layers
net.train_batch_norm()

# freeze BatchNorm layers for fine-tuning
net.freeze_batch_norm(freeze_running_mean_var=True, freeze_gamma_beta=True):

# freeze all trainable layers except BatchNorm layers for better running mean and variance value
net.precise_batch_norm()

# load pre-trained parameters
net.load_parameters(filename, map_location=None, strict=True)

# load pre-trained backbone parameters
net.load_backbone_parameters(filename, map_location=None, strict=True)

# reset segmentation classes
net.reset_classes(num_classes)
```

Loss API

```Python
# change OHEM ratio
criterion.set_ohem_ratio(ohem_ratio)
```

Metric API

```Python
# update metric
miou.update(pred, target)

# get Mean IoU w/ background
miou.get()

# get Mean IoU w/o background
miou.get(ignore_background=True)

# reset metric
miou.reset()
```
