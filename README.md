# Driving-Behavior-Cloning
Neural Network that mimics the driving behavior

#### The goals / steps of this project are the following:
* Use the driving simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around multiple track without leaving the road

#### This project includes the following files:
* behavior_cloning.py containing the script to train the model and use model for inference (if required)
* utils.py containing supplmentary python functions
* download_resources.sh shell file to download driving-simulator and default-training-dataset
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network

#### Driving the car autonomously 
driving-simulator-files, `drive.py` script and saved model file is required to run the car autonomously in the simulator 
driving-simulator-files would be downloaded if download_resources.sh script is ran.
1. `./linux_sim/linux_sim.x86_64`
 run this command to start the simulator, then select default options (as training data was recorded using the default settings)
2. `python3 drive.py -s=17 -vfd=./output`
  this command start a server that continously posts steering angle information to the simulator, also './output' is the directory where video frame are stored.
3. `start simulation`
  drive.py is waiting for simulation to start, select a track and click automonous mode. Now, one should see the car moving forward.
4. `python3 video.py ./output` this commands converts stored video frames to a video file

### Model Architecture and Training Strategy

#### Model Architecture
The following table summaries the stacked layers used to build behanvior-cloning-network
file utils.py method get_model(...)

| Layer              | Input  | Output | Params |
|:------             |:-------|:-------|:-------|
| Cropping2D         | (?, 160, 320, 3) | (?, 90, 320, 3) | 0 |
| Lambda             | (?, 90, 320, 3)  | (?, 66, 200, 3) | 0 |
| Lambda             | (?, 66, 200, 3)  | (?, 66, 200, 3) | 0 |
| Conv2D             | (?, 66, 200, 3)  | (?, 31, 98, 24) | 1824 |
| BatchNormalization | (?, 31, 98, 24)  | (?, 31, 98, 24) | 96 |
| Activation         | (?, 31, 98, 24)  | (?, 31, 98, 24) | 0 |
| Dropout            | (?, 31, 98, 24)  | (?, 31, 98, 24) | 0 |
| Conv2D             | (?, 31, 98, 24)  | (?, 14, 47, 36) | 21636 |
| BatchNormalization | (?, 14, 47, 36)  | (?, 14, 47, 36) | 144 |
| Activation         | (?, 14, 47, 36)  | (?, 14, 47, 36) | 0 |
| Dropout            | (?, 14, 47, 36)  | (?, 14, 47, 36) | 0 |
| Conv2D             | (?, 14, 47, 36)  | (?, 5, 22, 48)  | 43248 |
| BatchNormalization | (?, 5, 22, 48)   | (?, 5, 22, 48)  | 192 |
| Activation         | (?, 5, 22, 48)   | (?, 5, 22, 48)  | 0 |
| Dropout            | (?, 5, 22, 48)   | (?, 5, 22, 48)  | 0 |
| Conv2D             | (?, 5, 22, 48)   | (?, 3, 20, 64)  | 27712 |
| BatchNormalization | (?, 3, 20, 64)   | (?, 3, 20, 64)  | 256 |
| Activation         | (?, 3, 20, 64)   | (?, 3, 20, 64)  | 0 |
| Dropout            | (?, 3, 20, 64)   | (?, 3, 20, 64)  | 0 |
| Conv2D             | (?, 3, 20, 64)   | (?, 1, 18, 64)  | 36928 |
| BatchNormalization | (?, 1, 18, 64)   | (?, 1, 18, 64)  | 256 |
| Activation         | (?, 1, 18, 64)   | (?, 1, 18, 64)  | 0 |
| Dropout            | (?, 1, 18, 64)   | (?, 1, 18, 64)  | 0 |
| Flatten            | (?, 1, 18, 64)   | (?, 1152)       | 0 |
| Dense              | (?, 1152)        | (?, 100)        | 115300 |
| BatchNormalization | (?, 100)         | (?, 100)        | 400 |
| Activation         | (?, 100)         | (?, 100)        | 0 |
| Dropout            | (?, 100)         | (?, 100)        | 0 |
| Dense              | (?, 100)         | (?, 50)         | 5050 |
| Dense              | (?, 50)          | (?, 10)         | 510 |
| Dense              | (?, 10)          | (?, 1)          | 11 |

Total params: 253,563

Trainable params: 252,891

Non-trainable params: 672


![alt text][image1]

#### Avoiding overfitting
11111111111

#### Parameter tuning

#### Training methodology
11111111111111

#### Creation of the Training Set & Training Process
11111111111111111

### Architecture and Training Documentation

#### Architecture experimentation
111111111111111

#### Training data collection
After the collection process, I had X number of data points. I then preprocessed this data by ...
I finally randomly shuffled the data set and put Y% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


### Simulation

