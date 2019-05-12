## BLSeg (BaseLine Segmentation)

PyTorch's Semantic Segmentation Toolbox

### Requirement

* python 3.x
* pytorch >= 1.0.0

---

### Supported Module

* Backbone
  * [VGG16]
  * [MobileNet v1]
  * [MobileNet v2]
  * [ResNet50] (Modified according to [Bag of Tricks])
  * [Modified Aligned Xception]
* Model
  * [FCN]
  * [U-Net]
  * [PSPNet]
  * [DeepLab v3+]
* Loss
  * BCEWithLogitsLossWithOHEM
  * CrossEntropyLossWithOHEM
  * DiceLoss (only for binary classification)
* Metric
  * Pixel Accuracy
  * Mean IoU

Each model can choose any backbone without any modification

|       Backbone \ Model        | **FCN** | **U-Net** | **PSPNet** | **DeepLab v3+** |
| :---------------------------: | :-----: | :-------: | :--------: | :-------------: |
|           **VGG16**           | &radic; |  &radic;  |  &radic;   |     &radic;     |
|       **MobileNet v1**        | &radic; |  &radic;  |  &radic;   |     &radic;     |
|       **MobileNet v2**        | &radic; |  &radic;  |  &radic;   |     &radic;     |
|         **ResNet50**          | &radic; |  &radic;  |  &radic;   |     &radic;     |
| **Modified Aligned Xception** | &radic; |  &radic;  |  &radic;   |     &radic;     |

Model pre-trained on augmented PASCAL VOC2012 dataset with 10582 images for training and 1449 images for validation.

You can download pre-trained parameters at [Google Drive]

---

### Usage

Import BLSeg Package

```python
from blseg import model, loss, metric
```

Create Model

```python
num_classes = 21

net = model.FCN('vgg16', num_classes)
net = model.ModernUNet('mobilenetv1', num_classes)
net = model.PSPNet('resnet50', num_classes)
net = model.DeepLabV3Plus('xception', num_classes)
```

Create Loss

```python
ohem_ratio = 0.7

criterion = loss.BCEWithLogitsLossWithOHEM(ohem_ratio)
criterion = loss.CrossEntropyLossWithOHEM(ohem_ratio)
criterion = DiceLoss()
```

Create Metric

```python
pixacc = metric.PixelAccuracy()
miou = metric.MeanIoU(num_classes)
```

Model API

```python
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

Metric API

```python
# update metric
metric.update(pred, target)
# get current metric
metric.get()
# reset metric
metric.reset()
```

---

### Changelog

See [CHANGELOG]

---

### References

* Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
* Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." arXiv preprint arXiv:1704.04861 (2017).
* Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
* He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
* Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
* Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
* Zhao, Hengshuang, et al. "Pyramid scene parsing network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
* Chen, Liang-Chieh, et al. "Encoder-decoder with atrous separable convolution for semantic image segmentation." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
* Xie, Junyuan, et al. "Bag of tricks for image classification with convolutional neural networks." arXiv preprint arXiv:1812.01187 (2018).

---

[VGG16]:https://arxiv.org/abs/1409.1556
[MobileNet v1]:https://arxiv.org/abs/1704.04861
[MobileNet v2]:https://arxiv.org/abs/1801.04381
[ResNet50]:https://arxiv.org/abs/1512.03385
[FCN]:https://arxiv.org/abs/1411.4038
[U-Net]:https://arxiv.org/abs/1505.04597
[PSPNet]:https://arxiv.org/abs/1612.01105
[DeepLab v3+]:https://arxiv.org/abs/1802.02611
[Modified Aligned Xception]:https://arxiv.org/abs/1802.02611
[Bag of Tricks]:https://arxiv.org/abs/1812.01187
[CHANGELOG]:https://github.com/linbo0518/LLSeg/blob/master/CHANGELOG.md
[Google Drive]:https://drive.google.com/drive/folders/1i1vhf-JQ_K-5SzS7OJQ9ns3wHCEwoSuD?usp=sharing
