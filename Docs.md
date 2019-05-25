### Docs

* Note: The order of the color channels is **BGR**, please use `cv2.imread` to read image.
* Note: the **mean** is `(0.485, 0.456, 0.406)` and the **std** is `(0.229, 0.224, 0.225)`

Import BLSeg Package

```Python
from blseg import model, loss, metric
```

#### Initialize

Create Model

```Python
num_classes = 21

# Available backbone: vgg16, resnet50, mobilenetv1, mobilenetv2, xception
net = model.FCN('vgg16', num_classes)
net = model.ModernUNet('mobilenetv1', num_classes)
net = model.PSPNet('resnet50', num_classes)
net = model.DeepLabV3Plus('xception', num_classes)
```

Create Loss

```Python
ohem_ratio = 0.7

# Binary Cross Entropy Loss with OHEM
criterion = loss.BCEWithLogitsLossWithOHEM(ohem_ratio)

# Cross Entropy Loss with OHEM
criterion = loss.CrossEntropyLossWithOHEM(ohem_ratio)

# Dice Loss
criterion = loss.DiceLoss()

# Soft-label Cross Entropy Loss with OHEM
criterion = loss.SoftCrossEntropyLossWithOHEM(ohem_ratio)
```

Create Metric

```Python
# Pixel Accuracy
pixacc = metric.PixelAccuracy()

# Mean IoU
miou = metric.MeanIoU(num_classes)

# Mean IoU w/o background
miou_nobg = metric.MeanIoU(num_classes, ignore_background=True)
```

#### Usage

Model API

```Python
# train backbone (default is train the entire net)
net.train_backbone()

# freeze backbone, only train the segmentation head
net.freeze_backbone()

# freeze BatchNorm layers for fine-tuning
net.freeze_BN()

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

# get current metric
miou.get()

# reset metric
miou.reset()
```

---