#### 2019.04.07

* add [MobileNet v1](https://arxiv.org/abs/1704.04861) backbone to FCN
* fix minor bugs

---

#### 2019.04.04

* add [FCN](https://arxiv.org/abs/1411.4038)
* add VGG16 and ResNet34 backbone to FCN
* refactor ResNet34's _add_stage function to make it more flex

---

#### 2019.03.31

* use kaiming normal initializer for ResUNet
* add [DeepLab v3+](https://arxiv.org/abs/1802.02611)
* clean code

---

#### 2019.03.23

* change UNet in_ch=3 and add ResUNet
* add [ResNet34](https://arxiv.org/abs/1512.03385) backbone to unet
* add ResidualBlock and DecodeBlock

---

#### 2019.03.22

* update .gitignore
* del padding arg in conv3x3
* use kaiming normal initializer for unet

---

#### 2019.03.21

* first init
* add [U-Net](https://arxiv.org/abs/1505.04597)

---