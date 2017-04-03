# Behavioral Cloning Project

Overview
---
Deep neural networks and convolutional neural networks applied to clone driving behavior. Train, validate and test a model using Keras/TensorFlow. The model outputs a steering angle to an autonomous vehicle.

Udacity provided a simulator where we can steer a car around a track for data collection. Then we use image data and steering angles to train a neural network and then use the model to drive the car autonomously around the track.

**[Track1 Video Record, 'Fastest' graphical mode](https://vimeo.com/208941014)** Click on this link to watch the video recording in 'fastest' mode for track 1
**[Track2 Video Record, 'Fastest' graphical mode](https://vimeo.com/208941482)** Click on this link to watch the video recording in 'fastest' mode for track 2

The Project
---
The goals/steps of this project are the following:
* Use the simulator to collect data on good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report
* Track 2 is a challenge!


## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a WebSocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the basic file.

#### Saving a video of the autonomous agent

```
python drive.py model.h5 run1
```

The fourth argument `run1` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.

```
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```
python video.py run1
```

Create a video based on images found in the `run1` directory. The name of the video will be the name of the directory following by `'.mp4'`, so, in this case, the video will be `run1.mp4`.

Optionally one can specify the FPS (frames per second) of the video:

```
python video.py run1 --fps 48
```

The video will run at 48 FPS. The default FPS is 60.


## Acknowledgments / References
* [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/drive)
* [NVIDIA's End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
* [Mez Gebre](https://mez.github.io/deep%20learning/2017/02/14/mainsqueeze-the-52-parameter-model-that-drives-in-the-udacity-simulator/)
* [Vivek Yadav](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.z6krjwf9e)
* [Andrew Hogan](https://hoganengineering.wixsite.com/randomforest/single-post/2017/03/13/Alright-Squares-Lets-Talk-Triangles)
