import os
import pickle
import numpy as np

from keras.callbacks import EarlyStopping, TerminateOnNaN, TensorBoard, ReduceLROnPlateau
from keras.models import load_model
from sklearn.model_selection import train_test_split
from math import ceil
from utils import *

np.set_printoptions(precision = 4)

samples = []

add_samples(samples, "./data/custom_data_track_1", 0.100)
# this sample was collected by driving under 15mph
# while staying in center of lane
add_samples(samples, "./data/custom_data_track_2_1", 0.100)
# this sample was collected by driving as fast as possible
# while staying in center of lane (as far as possible)
add_samples(samples, "./data/custom_data_track_2_2", 0.175)

print("Number of samples : ", len(samples))

train_samples, validation_samples = train_test_split(samples, test_size = 0.25)
N_train = len(train_samples)
N_val = len(validation_samples)
print("Number of training samples   : ", N_train)
print("Number of validation samples : ", N_val)
print("After augmentation number of training samples   : ", 2 * N_train)
print("After augmentation number of validation samples : ", 2 * N_val)

_BATCH_SIZE = 32
train_generator = generator(train_samples, _BATCH_SIZE)
validation_generator = generator(validation_samples, _BATCH_SIZE)

model_base_dir = "./model"
model_chkpt_file = model_base_dir + "/checkpoint/model_chkpt.h5"
model_weights_file = model_base_dir + "/weights/model_weights.h5"
tb_logs_dir = model_base_dir + "/tb-logs"
train_hist_file = model_base_dir + "/training-history/train_history_dict.p"

print("\n\n")

history = None
model = None
_LOAD_MODEL = False

if _LOAD_MODEL:
    model = load_model(model_chkpt_file)
    print("Model loaded from disk")

    with open(train_hist_file, 'rb') as f:
        history = pickle.load(f)
    print("Training history loaded from disk")

else:
    _EPOCHS = 30
    model = get_model(_EPOCHS = _EPOCHS, μ = 0.0, σ = 5e-2,\
                        λ = 5e-4, α = 3e-3,\
                        conv_drop_rate = 0.15, fc_drop_rate = 0.65)
    callbacks = [
        ReduceLROnPlateau(monitor = "val_loss", factor = 0.1,
                          patience = 3,
                          verbose = 1,
                          min_delta = 0.0050,
                          min_lr = 1e-8),
        EarlyStopping(monitor = "val_loss", min_delta = 0.0050,
                      patience = 9, verbose = 1),
        TensorBoard(log_dir = tb_logs_dir, batch_size = _BATCH_SIZE),
        TerminateOnNaN()
    ]

    historyObj = model.fit_generator(train_generator,
                    steps_per_epoch = ceil((2 * N_train) / _BATCH_SIZE),
                    validation_data = validation_generator,
                    validation_steps = ceil((2 * N_val) / _BATCH_SIZE),
                    epochs = _EPOCHS, callbacks = callbacks,
                    shuffle = True, verbose = 1)
    history = historyObj.history

    os.system("spd-say 'Model Trained'")
    print("\n\n")
    model.summary()
    print("\n\n")

    model.save(model_chkpt_file)
    print("Model saved to disk")

    model.save_weights(model_weights_file)
    print("Model weights saved to disk")

    with open(train_hist_file, 'wb') as f:
        pickle.dump(history, f)
    print("Training history saved to disk")

print("\n\n")

plot_base_dir = "./plots"

draw_line_graphs(plot_base_dir + "/train_loss-vs-val_loss.png",
                history['val_loss'], y1_label = "Validation loss",
                y2 = history['loss'], y2_label = "Training loss",
                title = "Validation loss vs Training loss",
                xlabel = "Epoch(s)", ylabel = "Loss",
                legend_loc = "lower right")

os.system("spd-say 'Program executed!'")
