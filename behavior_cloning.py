import os
import csv
import pickle
import numpy as np

from keras.models import load_model
from math import ceil
from utils import *

np.set_printoptions(precision = 4)

data_set_folder = "./data/custom_data_track_1"
img_folder = data_set_folder + "/IMG"

csv_file = data_set_folder + "/driving_log.csv"
additional_steer = 0.10
samples = []
with open(csv_file) as csvfile:
    reader = csv.reader(csvfile)
    for row_num, line in enumerate(reader):
        if row_num == 0:
            continue
        for col_num in range(3):
            center_angle = float(line[3])
            image_file_path = img_folder + "/" + line[col_num].split("/")[-1]
            if col_num == 1: # left camera image
                center_angle += additional_steer
            elif col_num == 2: # right camera image
                center_angle -= additional_steer
            samples.append([image_file_path, center_angle])

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
    _EPOCHS = 64
    model = get_model(_EPOCHS = _EPOCHS, μ = 0.0, σ = 5e-2,\
                        λ = 5e-3, α = 1e-2,\
                        conv_drop_rate = 0.20, fc_drop_rate = 0.50)
    callbacks = [
        ReduceLROnPlateau(monitor = "val_loss", factor = 0.1,
                          patience = 4,
                          verbose = 1,
                          min_delta = 0.001,
                          min_lr = 1e-8),
        EarlyStopping(monitor = "val_loss", min_delta = 0.0001,
                      patience = 20, verbose = 1),
        # TensorBoard(log_dir = tb_logs_dir, batch_size = _BATCH_SIZE),
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

    with open(train_hist_file, 'wb') as f:
        pickle.dump(history, f)
    print("Training history saved")

print("\n\n")

plot_base_dir = "./plots"

draw_line_graphs(plot_base_dir + "/train_loss-vs-val_loss.png",
                history['val_loss'], y1_label = "Validation loss",
                y2 = history['loss'], y2_label = "Training loss",
                title = "Validation loss vs Training loss",
                xlabel = "Epoch(s)", ylabel = "Loss",
                legend_loc = "lower right")

os.system("spd-say 'Program executed!'")
