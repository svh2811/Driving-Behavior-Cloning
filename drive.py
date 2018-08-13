import argparse
import base64
from datetime import datetime
import os
import shutil
import tensorflow as tf
import atexit

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


def invalid_speed_exit_message(speed):
    print("Invalid speed value {%d}, speed value must be between 1 and 30"
                .format(speed))


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        throttle = controller.update(float(speed))

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.video_frames_directory != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.video_frames_directory, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        '-s', '--speed',
        type=int,
        default=15,
        help='Speed, should be a whole number between 1 and 30 inclusive.'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default="./model/checkpoint/model_chkpt.h5",
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        '-vfd', '--video_frames_directory',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    if args.speed < 1 or args.speed > 30:
        atexit.register(invalid_speed_exit_message, speed=args.speed)

    controller.set_desired(args.speed)

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    # custom_objects is required since any tensorflow function included in
    # model architecture will need tf handle for reference
    # this is a work around for presumably a limitation of keras
    # for this version
    model = load_model(args.model, custom_objects={"tf": tf})

    if args.video_frames_directory != '':
        print("Creating image folder at {}".format(args.video_frames_directory))
        if not os.path.exists(args.video_frames_directory):
            os.makedirs(args.video_frames_directory)
        else:
            shutil.rmtree(args.video_frames_directory)
            os.makedirs(args.video_frames_directory)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
