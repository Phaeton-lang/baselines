'''Comparing a simple CNN with a convolutional MoE model on the CIFAR10 dataset. Based on the cifar10_cnn.py file in the
keras/examples folder.
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from ConvolutionalMoE import Conv2DMoE
from DenseMoE import DenseMoE
#from scipy.io import savemat
import os
import argparse
from datetime import datetime
import time

parser = argparse.ArgumentParser()
# LMS parameters
#lms_group = parser.add_mutually_exclusive_group(required=False)
parser.add_argument('--lms', dest='lms', action='store_true', help='Enable LMS')
parser.add_argument('--no-lms', dest='lms', action='store_false', help='Disable LMS (Default)')
parser.add_argument('--batch_size', type=int, default=512, help="Batch size. (Default 512)")
parser.add_argument('--steps', type=int, default=3, help="Steps per epoch. (Default 3)")
parser.add_argument('--epochs', type=int, default=1, help="Training epochs. (Default 1)")
parser.add_argument('--experts', type=int, default=8, help="Num of experts. (Default 8)")
parser.set_defaults(lms=False)
args = parser.parse_args()

if args.lms:
    print('==> enable LMS!')
    tf.config.experimental.set_lms_enabled(True)
    tf.experimental.get_peak_bytes_active(0)

batch_size = args.batch_size
num_classes = 10
epochs = args.epochs
data_augmentation = True
num_predictions = 20
#which_model = 'cnn' # 'moe' or 'cnn'
which_model = 'moe' # 'moe' or 'cnn'
job_idx = 3

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

if which_model == 'moe':
    # MoE model
    num_experts_per_filter = args.experts
    model = Sequential()
    model.add(Conv2DMoE(32, num_experts_per_filter, (3, 3), expert_activation='relu', gating_activation='softmax', padding='same', input_shape=x_train.shape[1:]))
    model.add(Conv2DMoE(32, num_experts_per_filter, (3, 3), expert_activation='relu', gating_activation='softmax'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2DMoE(64, num_experts_per_filter, (3, 3), expert_activation='relu', gating_activation='softmax', padding='same'))
    model.add(Conv2DMoE(64, num_experts_per_filter, (3, 3), expert_activation='relu', gating_activation='softmax'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(DenseMoE(512, num_experts_per_filter, expert_activation='relu', gating_activation='softmax'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

elif which_model == 'cnn':
    # plain Conv model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

# initiate RMSprop optimizer
# FIXME: In tf2.0, this API is updated!
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    hist = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        fill_mode='nearest',  # set mode for filling points outside the input boundaries
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        rescale=None,  # set rescaling factor (applied before any other transformation)
        preprocessing_function=None,  # set function that will be applied on each input
        data_format=None  # image data format, either "channels_first" or "channels_last"
    )
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs=epochs,
                        #steps_per_epoch=len(x_train) / batch_size,
                        steps_per_epoch=args.steps,
                        validation_data=(x_test, y_test),
                        workers=4)
    print('peak active bytes(MB): {}'.format(tf.experimental.get_peak_bytes_active(0)/1024.0/1024.0))

# Score trained model.
#scores = model.evaluate(x_test, y_test, verbose=1)
#print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])
