# BLSeg (BaseLine Segmentation)

![love](https://img.shields.io/badge/💖-build%20with%20love-blue.svg?style=for-the-badge)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/linbo0518/BLSeg?style=for-the-badge)
![GitHub](https://img.shields.io/github/license/linbo0518/BLSeg?style=for-the-badge)

PyTorch's Semantic Segmentation Toolbox

- [BLSeg (BaseLine Segmentation)](#blseg-baseline-segmentation)
  - [Requirement](#requirement)
  - [Quick Start](#quick-start)
  - [Documentation](#documentation)
  - [Supported Module](#supported-module)
  - [Analysis](#analysis)
  - [Visualization](#visualization)
  - [Changelog](#changelog)
  - [References](#references)

## Requirement

- Python 3
- PyTorch >= 1.0.0

## Quick Start

Execute the following command in your terminal

```sh
pip install --upgrade git+https://github.com/linbo0518/BLSeg.git
```

## Documentation

For more information, please see [Documentation](Documentation.md)

## Supported Module

- Backbone
  - [VGG16]
  - [VGG19]
  - [MobileNet v1] (1.0)
  - [MobileNet v2] (1.0)
  - [ResNet 34]
  - [ResNet 50] (Modified according to [Bag of Tricks])
  - [SE ResNet 34]
  - [SE ResNet 50] (Modified according to [Bag of Tricks])
  - [Modified Aligned Xception]
- Model
  - [FCN]
  - [U-Net]
  - [PSPNet]
  - [DeepLab v3+]
  - [GCN] (Large Kernel Matters)
- Loss
  - BCEWithLogitsLossWithOHEM
  - CrossEntropyLossWithOHEM
  - DiceLoss (only for binary classification)
  - SoftCrossEntropyLossWithOHEM
- Metric
  - Loss Meter
  - Pixel Accuracy
  - Mean IoU
- Others
  - Xavier/MSRA initialization (support zero gamma in last BatchNorm)
  - Pre-trained weight
  - Mixed precision training
  - Online Hard Example Mining
  - Precise BatchNorm
  - Freeze/train Backbone model
  - Freeze/train BatchNorm layers
  
Each segmentation model can combine with any backbone without any modifications.

|       Backbone \ Model        | **FCN** | **U-Net** | **PSPNet** | **DeepLab v3+** | **GCN** |
| :---------------------------: | :-----: | :-------: | :--------: | :-------------: | :-----: |
|           **VGG16**           | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |
|           **VGG19**           | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |
|       **MobileNet v1**        | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |
|       **MobileNet v2**        | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |
|         **ResNet34**          | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |
|         **ResNet50**          | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |
|        **SE ResNet34**        | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |
|        **SE ResNet50**        | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |
| **Modified Aligned Xception** | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |

Model pre-trained on augmented PASCAL VOC2012 dataset with 10582 images for training and 1449 images for validation.

You can download pre-trained parameters at [Google Drive](https://drive.google.com/drive/folders/1i1vhf-JQ_K-5SzS7OJQ9ns3wHCEwoSuD?usp=sharing)

## Analysis

- Parameters

|       Backbone \ Model        | **FCN** | **U-Net** | **PSPNet** | **DeepLab v3+** | **GCN** |
| :---------------------------: | ------: | --------: | ---------: | --------------: | ------: |
|           **VGG16**           | 134.82M |    25.26M |     19.71M |          20.15M |  15.36M |
|           **VGG19**           | 140.13M |    30.57M |     25.02M |          25.46M |  20.67M |
|       **MobileNet v1**        | 226.07M |    14.01M |     13.71M |          12.44M |   4.04M |
|       **MobileNet v2**        | 276.46M |     2.68M |     15.67M |          13.36M |   2.88M |
|         **ResNet34**          | 141.38M |    24.08M |     26.28M |          26.72M |  21.76M |
|         **ResNet50**          | 451.92M |    66.35M |     46.61M |          40.37M |  25.09M |
|        **SE ResNet34**        | 141.54M |    24.25M |     26.44M |          26.87M |  21.92M |
|        **SE ResNet50**        | 454.44M |    69.03M |     49.13M |          42.89M |  27.61M |
| **Modified Aligned Xception** | 466.25M |    57.46M |     60.95M |          54.71M |  39.17M |

- Multiply-accumulate operations (MACs)

|       Backbone \ Model        | **FCN** | **U-Net** | **PSPNet** | **DeepLab v3+** | **GCN** |
| :---------------------------: | ------: | --------: | ---------: | --------------: | ------: |
|           **VGG16**           | 348.11G |   114.79G |    121.38G |         121.38G |  85.42G |
|           **VGG19**           | 390.27G |   136.54G |   103.21Gs |         127.44G | 107.17G |
|       **MobileNet v1**        | 228.52G |    37.83G |     52.41G |          52.41G |   8.24G |
|       **MobileNet v2**        | 240.77G |     3.22G |     58.00G |          58.00G |   5.80G |
|         **ResNet34**          | 230.69G |    34.99G |    109.88G |         109.88G |  23.65G |
|         **ResNet50**          | 326.75G |   133.01G |    178.64G |         178.64G |  29.44G |
|        **SE ResNet34**        | 230.70G |    35.00G |    109.90G |         109.90G |  23.66G |
|        **SE ResNet50**        | 326.81G |   133.05G |    178.72G |         178.72G |  29.47G |
| **Modified Aligned Xception** | 359.83G |    83.46G |    237.01G |         237.01G |  44.88G |

## Visualization

|       Original Image        |       Ground Truth        |                Ours                 |
| :-------------------------: | :-----------------------: | :---------------------------------: |
| ![4_image](img/4_image.png) | ![4_mask](img/4_mask.png) | ![4_pred_mask](img/4_pred_mask.png) |
| ![7_image](img/7_image.png) | ![7_mask](img/7_mask.png) | ![7_pred_mask](img/7_pred_mask.png) |
| ![9_image](img/9_image.png) | ![9_mask](img/9_mask.png) | ![9_pred_mask](img/9_pred_mask.png) |

## Changelog

See [Changelog](Changelog.md)

## References

- Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
- Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." arXiv preprint arXiv:1704.04861 (2017).
- Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
- He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- Xie, Junyuan, et al. "Bag of tricks for image classification with convolutional neural networks." arXiv preprint arXiv:1812.01187 (2018).
- Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
- Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
- Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
- Zhao, Hengshuang, et al. "Pyramid scene parsing network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
- Chen, Liang-Chieh, et al. "Encoder-decoder with atrous separable convolution for semantic image segmentation." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
- Peng, Chao, et al. "Large Kernel Matters--Improve Semantic Segmentation by Global Convolutional Network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

---

[VGG16]:https://arxiv.org/abs/1409.1556
[VGG19]:https://arxiv.org/abs/1409.1556
[MobileNet v1]:https://arxiv.org/abs/1704.04861
[MobileNet v2]:https://arxiv.org/abs/1801.04381
[ResNet 34]:https://arxiv.org/abs/1512.03385
[ResNet 50]:https://arxiv.org/abs/1512.03385
[SE ResNet 34]:https://arxiv.org/abs/1709.01507
[SE ResNet 50]:https://arxiv.org/abs/1709.01507
[Modified Aligned Xception]:https://arxiv.org/abs/1802.02611
[Bag of Tricks]:https://arxiv.org/abs/1812.01187

[FCN]:https://arxiv.org/abs/1411.4038
[U-Net]:https://arxiv.org/abs/1505.04597
[PSPNet]:https://arxiv.org/abs/1612.01105
[DeepLab v3+]:https://arxiv.org/abs/1802.02611
[GCN]:https://arxiv.org/abs/1703.02719
