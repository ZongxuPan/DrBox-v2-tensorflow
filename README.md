DRBox-v2
==
This code is the tensorflow implementation of DrBox-v2 which is an improved detector with rotatable boxes for target detection in remote sensing images, and is written by Zongxu Pan (mail:zxpan@mail.ie.ac.cn), Quanzhi An(anquanzhi16@mails.ucas.ac.cn) and Lei Liu.

Introduction
--
DRBox-v2 is a detector with rotatable boxes for target detection in remote sensing images.

The details of this method can be found in our following paper:

Quanzhi An, Zongxu Pan*, Lei Liu, and Hongjian You, "DRBox-v2: An improved detector with rotatable boxes for target detection in SAR images", IEEE Transactions on Geoscience and Remote Sensing, to be published.

The DOI of this paper is 10.1109/TGRS.2019.2920534.

Please cite this paper in your publications if it helps your research.


Data preparation
--
1 Data annotation

   >Every image has a corresponding annotation file('.rbox' file). Each line in the annotation file ('.rbox') represents a target in the image and consists of six annotation parameters, which are the coordinates of the horizontal and vertical directions of the center point, the length, width, category and angle of the target (the included angle between the principal axis of the RBox and the X-axis).
   >For example, the size of the following figure is (300,300), and the annotation information of the target is:
   
   ![](https://github.com/ZongxuPan/DrBox-v2-tensorflow/blob/master/figure1.png)
   
```
   180.087143 235.796040 63.747273 14.874003 1 25.974394
```

2 train.txt

   >The list of training data. Each line in the file represents a training image, which contains the name of the training image and the annotation information file.
   Examples:
```Bash
  001.jpg 001.jpg.rbox
  002.jpg 002.jpg.rbox
```

3 test.txt
   >The list of test data. Each line in the file represents a test image, which contains the name of the test image and the annotation information file.
   Examples:
```Bash
  test01.jpg test01.jpg.rbox
  test02.jpg test02.jpg.rbox
```

Data interface
--
1 TXT_DIR

  >The path to the file "train.txt" and "test.txt".

2 INPUT_DATA_PATH, TEST_DATA_PATH

  >The path to the training data or test data.

3 PRETRAINED_NET_PATH

  >The path to the pre-training model(vgg16.npy).
  The file 'vgg16.npy' is the pretrained model of VGG16.

4 SAVE_PATH

  >The path to save the model.

5 IM_HEIGHT, IM_WIDTH, IM_CDIM

  >The height, width and the channels of the training images; in order to utilize the pre-training model obtained by optical images, it is recommended to transform training data into the three-channel images.  

6 PRIOR_HEIGHTS, PRIOR_WIDTHS

  >The height, width of the prior Rbox.

Important parameter description
--
1 os.environ["CUDA_VISIBLE_DEVICES"]
  > Choose the GPU to train or test the model.

2 LOAD_PREVIOUS_POS
  > For each dataset, the network needs to select positive sample and encode them according to preset parameters of prior RBox. So when you use new dataset or change the parameters of prior RBox, LOAD_PREVIOUS_POS should be set to "False"; when the data and these hyperparameters are unchanged, the encoding does not need to be computed repeatedly, so LOAD_PREVIOUS_POS should be set to "True" for fast implementation.

3 USE_THIRD_LAYER
   > This parameter determines whether or not to use the multi-layer prediction. Set it to 1 if used, otherwise set it to 0.

4 FPN_NET
   > This parameter determines whether or not to use the FPN structure. Set it to 1 if used, otherwise set it to 0.

5 USE_FOCAL_LOSS and focal_loss_factor
   > "USE_FOCAL_LOSS" determines whether or not to use focal loss. Set "USE_FOCAL_LOSS" to 1 if used and focal_loss_factor is the factor of focal loss, otherwise set "USE_FOCAL_LOSS" to 0.

Build Your Own Dataset
--
For building your own dataset, please use the annotation tool at https://github.com/liulei01/DRBox.


Configuration
--
tensorflow1.2.1

python2.7

numpy1.13

cuda8


Example
--
1 When you want to train a model:

```Bash
  python Drbox.py --train
```

2 When you want to test the model:

```Bash
  python Drbox.py
```


Q & A
--
1 In this implementation, the height and the width of the image are set as 300. Which part of the codes I need to change if the height and/or the width of the input image are not 300?
  > Please set IM_HEIGHT and IM_WIDTH as the desired values. In addition, FEA_HEIGHT4 and FEA_WIDTH4 which represent the size of the feature map of conv4_3 are needed to be altered accordingly, so are FEA_HEIGHT3 and FEA_WIDTH3 which denote the size of the feature map of conv3_3.