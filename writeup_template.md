# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Model_Visualization.png "Model Visualization"
[image2]: ./examples/Center.jpg "Grayscaling"
[image3]: ./examples/Recovery_01.jpg "Recovery Image"
[image4]: ./examples/Recovery_02.jpg "Recovery Image"
[image5]: ./examples/Recovery_03.jpg "Recovery Image"
[image6]: ./examples/Normal.jpg "Normal Image"
[image7]: ./examples/Flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py ./model/model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths with 24, 36, 48, 64 (clone.py lines 130-155) 

The model includes RELU layers to introduce nonlinearity (code lines 130-151), and the data is normalized in the model using a Keras lambda layer (code line 109). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer and batch normalization layer in order to reduce overfitting (clone.py lines 137, 138, 143, 148, 153). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 132). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 159).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road to make the model learn the stratgy that keep the car at the center of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to design appropriate convolution neural network to learn steering information from each image to let the car run on the center of the road

My first step was to use a convolution neural network model similar to the nvidia end to end self-driving model, I thought this model might be appropriate because this model works on the real car and real road, so it should be appropriate to use on the simulator.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that the accuracy on the training data and validation data are very similar.

Then I add one batch normalization layer after flatten layer, and one dropout layer between full connected layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track on some curve road, to improve the driving behavior in these cases, I made increase some data, for example, flipped the images from center camera, images from left and right cameras.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (clone.py lines 108-126) consisted of a convolution neural network with the following layers and layer sizes, we use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers, then we follow the five convolutional layers with three fully connected layers leading to an output control value which is the inverse turning radius.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the side of the road to the center. These images show what a recovery looks like starting from left and right sides :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc

After the collection process, I had 32134 number of data points. I then preprocessed this data by splitting them to 80% training data and 20% validation data.


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by monitoring the training loss and validation loss for each epoch. I used an adam optimizer so the learning rate was not tuned manually.
