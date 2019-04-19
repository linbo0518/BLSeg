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