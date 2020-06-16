# Image domain conversion using CycleGAN
This example shows how to convert images from one domain into another using CycleGAN

CycleGAN is a GAN model that is generally used for the following purposes.

   -  Style transfer (images and paintings) 
   -  Season conversion 
   -  Day / night conversion 
   -  Object transformation 

The difference from Pix2Pix, which also perform image-image conversion, is that CycleGAN uses unsupervised learning, so there is no need for a paired image dataset.
In this example, even with unsupervised learning, you can see the model convert the images by understanding whether the fruit was a whole one or a cut one.

![result image](https://github.com/matlab-deep-learning/Image-domain-conversion-using-CycleGAN/raw/master/pics_for_doc/image_6.png)
![result image](https://github.com/matlab-deep-learning/Image-domain-conversion-using-CycleGAN/raw/master/pics_for_doc/image_7.png)

## **Requirements**
- [MATLAB](https://jp.mathworks.com/products/matlab.html)
- [Deep Learning Toolbox](https://jp.mathworks.com/products/deep-learning.html)
- [Image Processing Toolbox](https://jp.mathworks.com/products/image.html)
- [Parallel Computing Toolbox](https://jp.mathworks.com/products/parallel-computing.html)

MATLAB version should be R2019b and later 


## **Usage**
The repository provides the following files:

-	CycleGANExample.mlx — Example showing how to train the CycleGAN model
-	generator.m — Function to create a CycleGAN generator network
-	discriminator.m — Function to create a CycleGAN discriminator network
-	cycleGanImageDatastore.m — Datastore to prepare batches of images for training
-  cycleGAN_1000.mat -  Pretrained model that converts apples to oranges and vice-versa

To run, open CycleGANExample.mlx and run the script. You can train the model or use the pretrained model by setting the doTraining flag to false. 


# **Reference**
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)
 (Jun-Yan Zhu.etc, 2017)


Copyright 2019-2020 The MathWorks, Inc.
**[Download a free MATLAB trial for Deep Learning](https://www.mathworks.com/products/deep-learning.html)**

[![View Image domain conversion using CycleGAN on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://jp.mathworks.com/matlabcentral/fileexchange/76986-image-domain-conversion-using-cyclegan)
