# Copyright 2018, 2019. IBM All Rights Reserved.
'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

if len(sys.argv) > 1 and sys.argv[1] == 'ktk':
    print('Running with Keras team Keras')
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K
else:
    print('Running with TensorFlow Keras')
    from tensorflow.python import keras
    from tensorflow.python.keras.datasets import mnist
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense, Dropout, Flatten
    from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.python.keras import backend as K

from tensorflow_large_model_support import LMS
tf.logging.set_verbosity(tf.logging.INFO)


batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Keras callback for LMS
# This model does not require TFLMS to successfully run. If we do not
# specify specific tuning parameters to LMS, the auto tuning will determine
# that TFLMS is not needed and disable it.
lms_callback = LMS(swapout_threshold=1, swapin_groupby=0, swapin_ahead=1, sync_mode=3)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[lms_callback])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('./mnistsavedmodel.h5', include_optimizer=False)
