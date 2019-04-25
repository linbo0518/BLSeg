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
  * Coming soon...
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

---

### TODO

- [ ] add pre-trained params
- [ ] add dice loss 
- [ ] add ohem feature to loss
- [ ] add large kernel matters (global convultional net)
- [ ] add more model

---

### Changelog

See [CHANGELOG]

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