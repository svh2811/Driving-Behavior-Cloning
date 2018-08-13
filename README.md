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
2. `python3 drive.py ./model/checkpoint/model_chkpt.h5`
  this command start a server that continously posts steering angle information to the simulator
3. `start simulation`
  drive.py is waiting for simulation to start, select a track and click automonous mode. Now, one should see the car moving forward.

### Model Architecture and Training Strategy

#### Model Architecture
The following table summaries the stacked layers used to build behanvior-cloning-network

file utils.py method get_model(...)

![alt text][image1]

#### Training methodology
11111111111111

#### Creation of the Training Set & Training Process
11111111111111111

#### Approach
111111111111111

#### Training data collection
After the collection process, I had X number of data points. I then preprocessed this data by ...
I finally randomly shuffled the data set and put Y% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
