import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime
import time

parser = argparse.ArgumentParser()
# LMS parameters
#lms_group = parser.add_mutually_exclusive_group(required=False)
parser.add_argument('--lms', dest='lms', action='store_true',
                    help='Enable LMS')
parser.add_argument('--no-lms', dest='lms', action='store_false',
                    help='Disable LMS (Default)')
parser.add_argument('batch_size', type=int,  help="batch size, e.g., 256")
parser.add_argument('height_width', type=int,  help="dataset scale, e.g., 32")
parser.add_argument('steps', type=int,  help="training steps, e.g., 10")
parser.set_defaults(lms=False)
args = parser.parse_args()

if args.lms:
    tf.config.experimental.set_lms_enabled(True)
    tf.experimental.get_peak_bytes_active(0)

img_h, img_w = args.height_width, args.height_width

num_classes = 100 # 0 to 9 digits
num_features = img_h*img_w # 28*28

# Training parameters.
learning_rate = 0.01
training_steps = args.steps
batch_size = args.batch_size
display_step = 1

# Prepare MNIST data.
# x_train shape: (60000, 28, 28)
# y_train shape: (60000,)
# x_test shape: (10000, 28, 28)
# y_test shape: (10000,)
#from tensorflow.keras.datasets import mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_imgs = 10000
x_train = np.random.randn(num_imgs, img_h, img_w).astype(np.int32)
y_train = np.random.randn(num_imgs).astype(np.int32)

print('x_train shape: {}'.format(x_train.shape))
print('y_train shape: {}'.format(y_train.shape))

# Convert to float32.
x_train = np.array(x_train, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train = x_train.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train = x_train / 255.

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
shuffle_size = int(num_imgs / 10)
train_data = train_data.repeat().shuffle(shuffle_size).batch(batch_size).prefetch(1)

# Weight of shape [784, 10], the 28*28 image features, and total number of classes.
W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")
# Bias of shape [10], the total number of classes.
b = tf.Variable(tf.zeros([num_classes]), name="bias")

# Logistic regression (Wx + b).
def logistic_regression(x):
    # Apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(tf.matmul(x, W) + b)

# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred),1))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = logistic_regression(x)
        loss = cross_entropy(pred, y)

    # Compute gradients.
    gradients = g.gradient(loss, [W, b])
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [W, b]))

# miliseconds
print(datetime.now().timetz())
time_list = []
# time in ms
cur_time = int(round(time.time()*1000))
# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    next_time = int(round(time.time()*1000))
    time_list.append(next_time - cur_time)
    cur_time = next_time

    if step % display_step == 0:
        pred = logistic_regression(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
        print('peak active bytes(MB): {}'.format(tf.experimental.get_peak_bytes_active(0)/1024.0/1024.0))
        #print('bytes in use(MB): {}'.format(tf.experimental.get_bytes_in_use(0)/1024.0/1024.0))

print('throughput: {} ms!!!'.format(np.average(np.array(time_list))))
