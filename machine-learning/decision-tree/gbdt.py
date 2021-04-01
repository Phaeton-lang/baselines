import tensorflow as tf
import numpy as np
import copy

import argparse
from datetime import datetime
import time

'''
Gradient Boosted Decision Tree (GBDT).
Implement a Gradient Boosted Decision tree with TensorFlow to classify
handwritten digit images. This example is using the MNIST database of
handwritten digits as training samples (http://yann.lecun.com/exdb/mnist/).

REFs:
- <https://github.com/aymericdamien/TensorFlow-Examples>
'''

num_classes = 2 # Total classes: greater or equal to $23,000, or not (See notes below).
num_features = 13 # data features size.

# Training parameters.
max_steps = 2000
batch_size = 256
learning_rate = 1.0
l1_regul = 0.0
l2_regul = 0.1

# GBDT parameters.
num_batches_per_layer = 1000
num_trees = 10
max_depth = 4

parser = argparse.ArgumentParser()
# LMS parameters
lms_group = parser.add_mutually_exclusive_group(required=False)
lms_group.add_argument('--lms', dest='lms', action='store_true',
                       help='Enable LMS')
lms_group.add_argument('--no-lms', dest='lms', action='store_false',
                       help='Disable LMS (Default)')
parser.set_defaults(lms=False)
args = parser.parse_args()

if args.lms:
    tf.config.experimental.set_lms_enabled(True)
    tf.experimental.get_peak_bytes_active(0)

#from tensorflow.keras.datasets import boston_housing
#(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

with np.load('boston_housing.npz', allow_pickle=True) as f:
    x = f['x']
    y = f['y']
seed = 113
test_split = 0.2
np.random.seed(seed)
indices = np.arange(len(x))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]
x_train = np.array(x[:int(len(x) * (1 - test_split))])
y_train = np.array(y[:int(len(x) * (1 - test_split))])
#x_test = np.array(x[int(len(x) * (1 - test_split)):])
#y_test = np.array(y[int(len(x) * (1 - test_split)):])

print('x_train shape: {}'.format(x_train.shape))
print('y_train shape: {}'.format(y_train.shape))

# For classification purpose, we build 2 classes: price greater or lower than $23,000
def to_binary_class(y):
    for i, label in enumerate(y):
        if label >= 23.0:
            y[i] = 1
        else:
            y[i] = 0
    return y

y_train_binary = to_binary_class(copy.deepcopy(y_train))
#y_test_binary = to_binary_class(copy.deepcopy(y_test))

train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'x': x_train}, y=y_train_binary,
    batch_size=batch_size, num_epochs=None, shuffle=True)

#test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
#    x={'x': x_test}, y=y_test_binary,
#    batch_size=batch_size, num_epochs=1, shuffle=False)

#test_train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
#    x={'x': x_train}, y=y_train_binary,
#    batch_size=batch_size, num_epochs=1, shuffle=False)

# GBDT Models from TF Estimator requires 'feature_column' data format.
feature_columns = [tf.feature_column.numeric_column(key='x', shape=(num_features,))]

gbdt_classifier = tf.estimator.BoostedTreesClassifier(
    n_batches_per_layer=num_batches_per_layer,
    feature_columns=feature_columns,
    n_classes=num_classes,
    learning_rate=learning_rate,
    n_trees=num_trees,
    max_depth=max_depth,
    l1_regularization=l1_regul,
    l2_regularization=l2_regul
)

gbdt_classifier.train(train_input_fn, max_steps=max_steps)
print('peak active bytes(MB): {}'.format(tf.experimental.get_peak_bytes_active(0)/1024.0/1024.0))
print('bytes in use(MB): {}'.format(tf.experimental.get_bytes_in_use(0)/1024.0/1024.0))

