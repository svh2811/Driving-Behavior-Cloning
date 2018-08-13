import cv2
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras import Sequential, regularizers
from keras.layers import Cropping2D, Conv2D, Dense, Flatten, Dropout, Lambda, BatchNormalization, Activation
from keras.initializers import TruncatedNormal
from keras.optimizers import Adam

from sklearn.utils import shuffle

def add_samples(samples, data_set_folder, additional_steer):
    with open(data_set_folder + "/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row_num, line in enumerate(reader):
            if row_num == 0:
                continue
            for col_num in range(3):
                center_angle = float(line[3])
                image_file_path = data_set_folder + "/IMG/"\
                                    + line[col_num].split("/")[-1]
                if col_num == 1: # left camera image
                    center_angle += additional_steer
                elif col_num == 2: # right camera image
                    center_angle -= additional_steer
                samples.append([image_file_path, center_angle])


def bgrToRgb(bgrImage):
    return cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)


def generator(samples, batch_size):
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while True:
        # shuffle before start of each epoch
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                image = bgrToRgb(cv2.imread(batch_sample[0]))
                center_angle = float(batch_sample[1])
                images.append(image)
                angles.append(center_angle)
                images.append(np.fliplr(image))
                angles.append(-center_angle)

            X = np.array(images)
            y = np.array(angles)
            yield shuffle(X, y) # shuffle within a batch


def conv_layer(model, depth, kernel, stride, trunc_normal, conv_drop_rate):
    model.add(Conv2D(depth, (kernel, kernel),
            kernel_initializer = trunc_normal,
            strides = (stride, stride)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(conv_drop_rate))


def dropout_dense_layer(model, out, trunc_normal, fc_drop_rate):
    model.add(Dense(out, kernel_initializer = trunc_normal))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(fc_drop_rate))


def l2_reg_dense_layer(model, out, trunc_normal, λ):
    model.add(Dense(out, kernel_initializer = trunc_normal,
                    kernel_regularizer = regularizers.l2(λ),
                    activation = "relu"))


def get_model(_EPOCHS, μ, σ, λ, α, conv_drop_rate, fc_drop_rate):
    tn = TruncatedNormal(mean = μ, stddev = σ)

    model = Sequential()
    model.add(Cropping2D(cropping = ((48, 22), (0, 0)),
                         input_shape = (160, 320, 3))) # (?, 90, 320, 3)
    model.add(Lambda(lambda x: tf.image.resize_images(x, (66, 200)))) # (?, 66, 200, 3)
    model.add(Lambda(lambda x: (x / 255.0) - 0.5)) # (?, 90, 320, 3)
    conv_layer(model, 24, 5, 2, tn, conv_drop_rate) # (?, 31, 98, 24)
    conv_layer(model, 36, 5, 2, tn, conv_drop_rate) # (?, 14, 47, 36)
    conv_layer(model, 48, 5, 2, tn, conv_drop_rate) # (?, 5, 22, 48)
    conv_layer(model, 64, 3, 1, tn, conv_drop_rate) # (?, 3, 20, 64)
    conv_layer(model, 64, 3, 1, tn, conv_drop_rate) # (?, 1, 18, 64)
    model.add(Flatten()) # (?, 1152)
    dropout_dense_layer(model, 100, tn, fc_drop_rate) # (?, 100)
    l2_reg_dense_layer(model, 50, tn, λ) # (?, 50)
    l2_reg_dense_layer(model, 10, tn, λ) # (?, 10)
    model.add(Dense(1, kernel_initializer = tn)) # (?, 1)
    model.compile(optimizer = Adam(lr = α), loss = "mean_squared_error")
    return model


def draw_line_graphs(fname,
                    y1, y1_label = "",
                    y2 = None, y2_label = "",
                    title = "", xlabel = "", ylabel = "",
                    legend_loc = "best"):
    plt.figure(figsize = (16, 9))
    plt.tight_layout()
    x_s = np.arange(len(y1))
    plt.plot(x_s, y1, 'b.-', label = y1_label)
    if y2 is not None:
        plt.plot(x_s, y2, 'r.-', label = y2_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    leg = plt.legend(loc = legend_loc)
    leg.get_frame().set_alpha(0.5)
    plt.grid(True)
    plt.savefig(fname = fname)
