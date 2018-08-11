import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras import Sequential, regularizers
from keras.layers import Cropping2D, Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout, Lambda, BatchNormalization, Activation
from keras.initializers import TruncatedNormal
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TerminateOnNaN, TensorBoard, ReduceLROnPlateau, LearningRateScheduler


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
    model.add(Cropping2D(cropping = ((50, 20), (0, 0)),
                         input_shape = (160, 320, 3))) # (?, 90, 320, 3)
    model.add(Lambda(lambda x: (x / 255.0) - 0.5)) # (?, 90, 320, 3)
    conv_layer(model, 24, 5, 2, tn, conv_drop_rate) # (?, 43, 158, 24)
    conv_layer(model, 36, 5, 2, tn, conv_drop_rate) # (?, 20, 77, 36)
    conv_layer(model, 48, 5, 2, tn, conv_drop_rate) # (?, 8, 37, 48)
    conv_layer(model, 64, 3, 1, tn, conv_drop_rate) # (?, 6, 35, 64)
    conv_layer(model, 96, 3, 1, tn, conv_drop_rate) # (?, 4, 33, 96)
    conv_layer(model, 128, 3, 1, tn, conv_drop_rate) # (?, 2, 31, 128)
    model.add(Flatten()) # (?, 3968)
    # 3968 / x^(4) = 1 ---> x = 7.93
    dropout_dense_layer(model, 500, tn, fc_drop_rate) # (?, 500)
    dropout_dense_layer(model, 63, tn, fc_drop_rate) # (?, 63)
    l2_reg_dense_layer(model, 8, tn, λ) # (?, 8)
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
