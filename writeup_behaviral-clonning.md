# **Behavioral Cloning** 

## Convolutional Neural Network with Keras/TensorFlow

### Join me on this exciting journey to build, train and validate a new deep neural network to clone driving behavior. The model outputs a steering angle to an autonomous vehicle! Thanks to Udacity Self-driving Car Nanodegree for providing me the basic skills set to get there!

#### A simulator where you can steer a car around a track for data collection has been provided by Udacity. I have used image data and steering angles to train the neural network and then used the model to drive the car autonomously around the track.

---

**Behavioral Cloning Project**

The goals/steps of this project are the following:
* Use the simulator to collect data on good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./my_images/nvidia.png "NVIDIA Model"
[image2]: ./my_images/my_nvidia.png "Implemented Model"
[image3]: ./my_images/tr1_model_summary.png "Model Summary"
[image4]: ./my_images/1-tr1_initial_distribution.png "Track 1 Initial Distribution"
[image5]: ./my_images/2-tr1_s0_drop.png "Track 1 S=0 Drop"
[image6]: ./my_images/3a-tr1_left_right_angle_correction.png "Track 1 Left and Right Angle Corrections"
[image7]: ./my_images/3b-tr1_left_right_angle_correction_images.png "Track 1 Images for Angle Corrections"
[image8]: ./my_images/4-tr1_flipping.png "Track 1 Flipping Images Driver Log Plan"
[image9]: ./my_images/5-tr1_horz_shift.png "Track 1 Horizontal Shift Images Driver Log Plan"
[image10]: ./my_images/6-tr1_drop_outbound_angles.png "Track 1 Outbound Drop"
[image11]: ./my_images/7-tr1_final_trimming.png "Track 1 Final Trimming"
[image12]: ./my_images/8-tr1_crop_image.png "Crop and Resize Images"
[image13]: ./my_images/9-tr1_random_brightness.png "Random Brightness Augmentation"
[image14]: ./my_images/10-tr1_random_shadow.png "Random Shadow Augmentation"
[image15]: ./my_images/11-tr1_final_image.png "Final Preprocessed Image"
[image16]: ./my_images/1-tr2_initial_distribution.png "Track 2 Initial Distribution"
[image17]: ./my_images/2-tr2_s0_drop.png "Track 2 S=0 Drop"
[image18]: ./my_images/3a-tr2_left_right_angle_correction.png "Track 2 Left and Right Angle Corrections"
[image19]: ./my_images/3b-tr2_left_right_angle_correction_images.png "Track 2 Images for Angle Corrections"
[image20]: ./my_images/4-tr2_flipping.png "Track 2 Flipping Images Driver Log Plan"
[image21]: ./my_images/6-tr2_drop_outbound_angles.png "Track 2 Outbound Drop"
[image22]: ./my_images/7-tr2_final_trimming.png "Track 2 Final Trimming"
[image23]: ./my_images/tr1_training_loss.png "Track 1 Training Loss"
[image24]: ./my_images/tr2a_training_loss.png "Track 2 Training Loss, Part 1"
[image25]: ./my_images/tr2b_training_loss.png "Track 1 Training Loss, Part 2"


---
## Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model_track1.h5** containing a trained convolution neural network for track 1 
* **model_track2b.h5** containing a trained convolution neural network  for track 2 (Challenge)
* **preprocess_util.py** containing preprocessing helpers functions
* **Playground-Track1.ipynb** containing the preprocessing of driver log before training and training tests for track 1
* **Playground-JungleTrack2.ipynb** containing the preprocessing of driver log before training and training tests for track 2
* **[Track1 Video Record, 'Fastest' graphical mode](https://vimeo.com/208941014)** Click on this link to watch the video recording in 'fastest' mode for track 1
* **[Track2 Video Record, 'Fastest' graphical mode](https://vimeo.com/208941482)** Click on this link to watch the video recording in 'fastest' mode for track 2
* **writeup_report.md** summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```
If running in 'fastest' mode, you can increase target speed in **drive.py** up to:
* 30MPH, track 1,  model_track1.h5
* 16MPH, track 2, model_track2b.h5

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model replicates [NVIDIA's End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). Check on my code (model.py lines 44-65). The model includes data normalization/zero-mean by 255/-0.5 using a Keras lambda layer, 5x5 and 3x3 convolutions using Keras Convolution2D, RELU layers to introduce nonlinearity, fully connected layers using Keras Flatten and Dense, and overfitting control using Keras Dropout. The loss is compiled using mean square error (mse) and adam optimizer.

Here is the NVIDIA's paper architecture:

![alt text][image1]

Here is a visualization of the implemented architecture:

![alt text][image2]

Here is the model summary:

![alt text][image3]


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 55, 57, 59). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 108-110). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 63).

#### 4. Appropriate training data

For track 1, I collected a minimum dataset of images, initially with about 10K but after the initial 95% cutoff of high-frequency steering angle = 0, the remaining dataset had 3.3K images only. I did a further augmentation and increased it up to 12.5K I was surprised by the achievements with such small dataset. 

Track 1:
* I collected my data from 'fastest' mode from Udacity simulator, which does not include shadows on the road.
* I collected images for 3 full cycles driving in the correct direction only. So without proper augmentation, the dataset would be biased for curves on one side.
* I used the keyboard arrows to drive, so the data is not smooth and there was a lot of s=0 because the key arrow is not held to keep constant steering angle.
* I tried to keep the car in the center all the time.
* I did not collect data with the car recovering itself from off the road. That could be future data improvements.

On the other hand, track 2 was a challenging one. I collected a bigger dataset of images, initially with about 53K, after the initial 95% cutoff of high-frequency steering angle = 0, the remaining dataset still had 49.5K images. I did further augmentation and data selection and ended up with a 46.6K dataset. 

Track 2:
* I collected my data from 'good' mode from Udacity simulator.
* I collected images for 2 full cycles driving in the correct direction and then 2 full cycles in the reverse direction.
* I used the mouse to steer, trying to get smooth data.
* I tried to keep the car in the right lane all the time, I am not a good racer so in order to achieve that I had to drive the car at 5MPH and bellow speed which I know would be a challenge to train the model and later use it on much higher speed driving.
* I did not collect data with the car recovering itself off the road. That could be future data improvements.
* I did not collect data with the car recovering itself from left to right lane. That could be future data improvements.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The main strategy was to make sure I had a well balanced and not biased dataset, then apply a well know model architecture such as [NVIDIA's End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

My first step was to evaluate the driver log steering histograms for 100 bins and do all required transformation, drop and augmentation to balance it. Please look at Playground notebooks for further details.

Track 1 Initial distribution:

![alt text][image4]

Track 1 S=0 95% drop:

![alt text][image5]

Track 1 Left and right steering angle correction:

![alt text][image6]
![alt text][image7]

Track 1 100% Image Flipping:

![alt text][image8]

Track 1 Horizontal shift:

![alt text][image9]

Track 1 Outbound angles drop:

![alt text][image10]

Track 1 Final trimming:

![alt text][image11]


The second step was to take care of images transformation as defined on the new driver log. The images also needed to be cropped to get rid of landscape and car's hood (image noise) and resize image to same as used by NVIDIA NN Model. Please look at Playground notebooks for further details. (model.py lines 74-81, preprocess_util.py all lines)

Track 1 Cropping image size:

![alt text][image12]

Track 1 Apply random brightness:

![alt text][image13]

Track 1 Apply random shadows:

![alt text][image14]

Track 1 Combined image transformation:

![alt text][image15]


Here is the track 2 driver log balance process:

Track 2 Initial distribution:

![alt text][image16]

Track 2 S=0 95% drop:

![alt text][image17]

Track 2 Left and right steering angle correction:

![alt text][image18]
![alt text][image19]

Track 2 40% Image Flipping:

![alt text][image20]

Track 1 Outbound angles drop:

![alt text][image21]

Track 1 Final trimming:

![alt text][image22]



#### 2. Training results

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80/20, model.py lines 83-85).

For track 1, after some experimentation, I noted that a loss of about 0.02 for train and validation, in general, would start to diverge (overfitting), so I have used an early stop function (model.py lines 26-42).  Trained 5 epochs. The model worked great on 'fastest' mode of the Udacity simulator, even up to 30MPH speed target!! I let it run several complete cycles to be sure it was a stable solution. I have tried the 'fantastic' mode of simulator and it did great as well but presented some 'drunk' drive at high speeds, but it managed to stay on the road all the time, the shadow augmentation really helped (remembering I collected data without shadows) for this mode since the detail shadows projected on the road did not interfered much with the driving experience.

Track 1 Loss training history:

![alt text][image23]


For track 2, I noted that a loss of about 0.03~0.05 for train and validation, in general, would start to diverge (overfitting), so I have used an early stop function (model.py lines 26-42). For the final model, I started setting up an early stop at 0.05 loss, which barely allowed 2 epochs train. I saved the weights of this model and tested it. It did not work very well getting the car eventually stuck on curves. Then I set up a smaller early stop at 0.04 loss. Trained 3 more epochs. The model worked great on 'fastest' mode of the Udacity simulator, even up to 16MPH speed target which surprised me since the data was recorded at 5MPH and lower!! Stayed in the right lane about 99% of the time!! I let it run several complete cycles to be sure it was a stable solution. But on more detailed graphical modes the car turn to get eventually stuck or changed to the left lane. The relatively small dataset I collected does not include recovering steering from left to the right lane, so the car could not adjust itself back on the right lane and eventually would get stuck somewhere. Then I tried to further improve the training with more epochs, set up a smaller early stop at 0.03 loss and trained 3 more epochs. It did not work, the car was not driving properly even on 'fastest' graphical mode of the Udacity simulator. So my final solution is the second training above described. Any future addition of data for off-road and left to right lane recovery would improve the driving experience in higher graphical modes of the simulator. I strongly believe that a better dataset recorded at the right range of speed (I did it at too slow MPH) will generate better driving cloning, but for now, I got a good solution for the challenge!!

Track 1 Loss training history:

![alt text][image24]
![alt text][image25]


# Acknowledgments / References
* [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/drive)
* [NVIDIA's End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
* [Mez Gebre](https://mez.github.io/deep%20learning/2017/02/14/mainsqueeze-the-52-parameter-model-that-drives-in-the-udacity-simulator/)
* [Vivek Yadav](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.z6krjwf9e)
* [Andrew Hogan](https://hoganengineering.wixsite.com/randomforest/single-post/2017/03/13/Alright-Squares-Lets-Talk-Triangles)
