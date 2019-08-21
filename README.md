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

python2.7.12

numpy1.13.0

scipy0.19.1

ctypes1.1.0

cuda8.0.61


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

3 When you want to evaluate the test results:

```Bash
  python evaluation.py
```


Q & A
--
1 In this implementation, the height and the width of the image are set as 300. Which part of the codes I need to change if the height and/or the width of the input image are not 300?
  > Please set IM_HEIGHT and IM_WIDTH as the desired values. In addition, FEA_HEIGHT4 and FEA_WIDTH4 which represent the size of the feature map of conv4_3 are needed to be altered accordingly, so are FEA_HEIGHT3 and FEA_WIDTH3 which denote the size of the feature map of conv3_3.

2 When I use my own data, the results are not satisfactory, how to check the problem and what can be done for improving the results?
  > The following aspects are suggested:
  >
  > (1) The two parameters PRIOR_HEIGHTS and PRIOR_WEIGHTS appear in pairs. In other words, the two lists should have the same length. Reasonable setting of these two parameters needs to be considered in combination with your own data. Generally, several distribution centers of the target lengths and widths in the data set are selected as the basis for setting these two parameters. Please see our paper "DRBox-v2:An Improved Detector with Rotatable Boxes for Target Detection in SAR Images" IV, B, 2) for reference.
  >
  > (2) The setting of the "IS180" parameter is related to the characteristics of the targets in your data. When the head and tail of the target can be distinguished, (the angles of targets ranges from 0 to 360 in the annotation file) the parameter should be set to False, and PRIOR_ANGLES can be set to [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]; otherwise, the parameter is set to True.
  >
  > (3) When you use your own data, we recommend that you still set the image block size to 300×300, so you don't need to change the feature size (FEA_HEIGHT and FEA_WIDTH) and some other parameters. Otherwise, you need to adjust the feature size, stepsize according to your own data. In addition, please ensure that your annotation data is consistent with the sample and there are no problems such as crossing the boundary.
  >
  > (4) In order to ensure the correctness of the data preparation, you could check your own annotation data as follows. You can mark the prior boxes in "pkl" file (which plays as positive samples) on images. The plotted box should cover the target well.
  >
  > (5) You can modify the parameter settings as needed:
  >
  >     a) OVERLAP_THRESHOLD: this parameter affects the selection of positive samples.
  >
  >     b) TEST_SCORE_THRESHOLD: this parameter controls retained results in the test phase.
  >
  >     c) TEST_NMS_THRESHOLD: this parameter controls the NMS threshold in the test phase.
  >
  > (6) You can check the loss curve during the training process. When the loss curve is abnormal, the following actions are recommended:
  >
  >     a) You can comment on one loss, and observe the situation of the other loss to locate the problem. (self.loss = self.loc_loss + self.conf_loss)
  >
  >     b) When the location loss is abnormal, there may be a problem with the positive sample coding. Please check the parameter setting and the data preparation. (please see the above (3), (4), and (5) for reference)
  >
  >     c) Please note that the unreasonable setting of PRIOR_HEIGHTS and PRIOR_WEIGHTS probably cause the abnormal of the loss.
  