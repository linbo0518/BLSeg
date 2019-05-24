#### 2019.05.24

* add set_ohem_ratio to loss with ohem
* add SoftCrossEntropyLossWithOHEM

---

#### 2019.05.12

* add pre-trained parameters
* add reset_classes function to SegBaseModule
* add usage
* add model and metric API docs
* minor bug fix

---

#### 2019.05.06

* minor bug fix
* add BCEWithLogitsLossWithOHEM and CrossEntropyLossWithOHEM
* add DiceLoss
* metric now can work in both binary classification and multi-class classification

---

#### 2019.04.30

* [MobileNetV1] change relu6 to relu
* [FCN] change fcn8s to fcn16s
* [PSPNet] modified by caffe's implementation

---

#### 2019.04.25

* seg model now has num_classes member
* add Metric subpackage 
  * add Pixel Accuracy metric
  * add Mean IoU metric
* add freeze_backbone and train_backbone func to seg model
* add load_parameters to backbone model
* add load_parameters and load_backbone_parameters to seg model

---

#### 2019.04.20

* add MobileNet v2 backbone
* update init_params
* fix minor bugs

---

#### 2019.04.13

* all backbone now based on BackboneBaseModule
* all model now based on SegBaseModule
* add PSPNet

---
 
#### 2019.04.12

* rename package name to blseg(baseline segmentation)
* xception now is a public backbone
* make backbone's output stride changeable (16 and 32)
* add xception to fcn and unet
* make deeplab v3 plus more flex

---

#### 2019.04.10

* packaging llseg
* add backbone sub package
* change from ResNet34 to ResNet50S
* fcn reborn
* unet reborn

---

#### 2019.04.07

* add MobileNetV1 backbone to FCN
* fix minor bugs

---

#### 2019.04.04

* add FCN
* add VGG16 and ResNet34 backbone to FCN
* refactor ResNet34's _add_stage function to make it more flex

---

#### 2019.03.31

* use kaiming normal initializer for ResUNet
* add DeepLab v3+
* clean code

---

#### 2019.03.23

* change UNet in_ch=3 and add ResUNet
* add ResNet34 backbone to unet
* add ResidualBlock and DecodeBlock

---

#### 2019.03.22

* update .gitignore
* del padding arg in conv3x3
* use kaiming normal initializer for unet

---

#### 2019.03.21

* first init
* add U-Net

---