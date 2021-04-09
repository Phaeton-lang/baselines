from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

import argparse
from datetime import datetime
import time

"""
Deep Convolutional Generative Adversarial Network Example.
Build a deep convolutional generative adversarial network (DCGAN) to
generate digit images from a noise distribution with TensorFlow v2.

REFs: https://github.com/aymericdamien/TensorFlow-Examples
"""

parser = argparse.ArgumentParser()
# LMS parameters
#lms_group = parser.add_mutually_exclusive_group(required=False)
parser.add_argument('--lms', dest='lms', action='store_true', help='Enable LMS')
parser.add_argument('--no-lms', dest='lms', action='store_false', help='Disable LMS (Default)')
parser.add_argument('batch_size', type=int,  help="batch size, e.g., 256")
parser.add_argument('height_width', type=int,  help="dataset scale, e.g., 32")
parser.add_argument('steps', type=int,  help="training steps, e.g., 10")
parser.set_defaults(lms=False)
args = parser.parse_args()

if args.lms:
    tf.config.experimental.set_lms_enabled(True)
    tf.experimental.get_peak_bytes_active(0)

# MNIST Dataset parameters.
# data features (img shape: 28*28).
#img_h, img_w = 28, 28
#img_h, img_w = 32, 32
img_h, img_w = args.height_width, args.height_width
#img_h, img_w = 224, 224
num_features = img_h*img_w

# Training parameters.
lr_generator = 0.0002
lr_discriminator = 0.0002
training_steps = args.steps
batch_size = args.batch_size
display_step = 1

# Network parameters.
noise_dim = 100 # Noise data points.

"""
# Prepare MNIST data.
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
num_imgs = 5000
x_train = np.random.randn(num_imgs, img_h, img_w).astype(np.float32)
y_train = np.random.randn(num_imgs).astype(np.float32)

# x_train shape: (60000, 28, 28)
# y_train shape: (60000,)
print('x_train shape: {}'.format(x_train.shape))
print('y_train shape: {}'.format(y_train.shape))

# Convert to float32.
x_train = np.array(x_train, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_train = x_train / 255.


# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
shuffle_size = int(num_imgs/10)
train_data = train_data.repeat().shuffle(shuffle_size).batch(batch_size).prefetch(1)

# Create TF Model.
class Generator(Model):
    # Set layers.
    def __init__(self):
        super(Generator, self).__init__()
        if img_h == 28:
            self.fc1 = layers.Dense(7 * 7 * 128)
        elif img_h == 32:
            self.fc1 = layers.Dense(8 * 8 * 128)
        elif img_h == 128:
            self.fc1 = layers.Dense(32 * 32 * 128)
        elif img_h == 224:
            self.fc1 = layers.Dense(56 * 56 * 128)
        elif img_h == 300:
            self.fc1 = layers.Dense(75 * 75 * 128)
        elif img_h == 296:
            self.fc1 = layers.Dense(74 * 74 * 128)
        self.bn1 = layers.BatchNormalization()
        self.conv2tr1 = layers.Conv2DTranspose(64, 5, strides=2, padding='SAME')
        self.bn2 = layers.BatchNormalization()
        self.conv2tr2 = layers.Conv2DTranspose(1, 5, strides=2, padding='SAME')

    # Set forward pass.
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.bn1(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        if img_h == 28:
            # New shape: (batch, 7, 7, 128)
            x = tf.reshape(x, shape=[-1, 7, 7, 128])
        elif img_h == 32:
            # New shape: (batch, 8, 8, 128)
            x = tf.reshape(x, shape=[-1, 8, 8, 128])
        elif img_h == 128:
            # New shape: (batch, 32, 32, 128)
            x = tf.reshape(x, shape=[-1, 32, 32, 128])
        elif img_h == 224:
            # New shape: (batch, 56, 56, 128)
            x = tf.reshape(x, shape=[-1, 56, 56, 128])
        elif img_h == 300:
            # New shape: (batch, 75, 75, 128)
            x = tf.reshape(x, shape=[-1, 75, 75, 128])
        elif img_h == 296:
            # New shape: (batch, 74, 74, 128)
            x = tf.reshape(x, shape=[-1, 74, 74, 128])
        # Deconvolution, image shape: (batch, 14, 14, 64)
        # Deconvolution, image shape: (batch, 16, 16, 64)
        # Deconvolution, image shape: (batch, 64, 64, 64)
        # Deconvolution, image shape: (batch, 112, 112, 64)
        # Deconvolution, image shape: (batch, 150, 150, 64)
        # Deconvolution, image shape: (batch, 148, 148, 64)
        x = self.conv2tr1(x)
        x = self.bn2(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # Deconvolution, image shape: (batch, 28, 28, 1)
        # Deconvolution, image shape: (batch, 32, 32, 1)
        # Deconvolution, image shape: (batch, 128, 128, 1)
        # Deconvolution, image shape: (batch, 224, 224, 1)
        # Deconvolution, image shape: (batch, 300, 300, 1)
        # Deconvolution, image shape: (batch, 296, 296, 1)
        x = self.conv2tr2(x)
        x = tf.nn.tanh(x)
        return x

# Generator Network
# Input: Noise, Output: Image
# Note that batch normalization has different behavior at training and inference time,
# we then use a placeholder to indicates the layer if we are training or not.
class Discriminator(Model):
    # Set layers.
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(64, 5, strides=2, padding='SAME')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(128, 5, strides=2, padding='SAME')
        self.bn2 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024)
        self.bn3 = layers.BatchNormalization()
        self.fc2 = layers.Dense(2)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, img_h, img_w, 1])
        x = self.conv1(x)
        x = self.bn1(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn3(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        return self.fc2(x)

# Build neural network model.
generator = Generator()
discriminator = Discriminator()

# Losses.
def generator_loss(reconstructed_image):
    gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=reconstructed_image, labels=tf.ones([batch_size], dtype=tf.int32)))
    return gen_loss

def discriminator_loss(disc_fake, disc_real):
    disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=disc_real, labels=tf.ones([batch_size], dtype=tf.int32)))
    disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=disc_fake, labels=tf.zeros([batch_size], dtype=tf.int32)))
    return disc_loss_real + disc_loss_fake

# Optimizers.
optimizer_gen = tf.optimizers.Adam(learning_rate=lr_generator)#, beta_1=0.5, beta_2=0.999)
optimizer_disc = tf.optimizers.Adam(learning_rate=lr_discriminator)#, beta_1=0.5, beta_2=0.999)

# Optimization process. Inputs: real image and noise.
def run_optimization(real_images):
    # Rescale to [-1, 1], the input range of the discriminator
    real_images = real_images * 2. - 1.
    # Generate noise.
    noise = np.random.normal(-1., 1., size=[batch_size, noise_dim]).astype(np.float32)
    with tf.GradientTape() as g:
        fake_images = generator(noise, is_training=True)
        disc_fake = discriminator(fake_images, is_training=True)
        disc_real = discriminator(real_images, is_training=True)
        disc_loss = discriminator_loss(disc_fake, disc_real)

    # Training Variables for each optimizer
    gradients_disc = g.gradient(disc_loss,  discriminator.trainable_variables)
    optimizer_disc.apply_gradients(zip(gradients_disc,  discriminator.trainable_variables))

    # Generate noise.
    noise = np.random.normal(-1., 1., size=[batch_size, noise_dim]).astype(np.float32)
    with tf.GradientTape() as g:
        fake_images = generator(noise, is_training=True)
        disc_fake = discriminator(fake_images, is_training=True)
        gen_loss = generator_loss(disc_fake)

    gradients_gen = g.gradient(gen_loss, generator.trainable_variables)
    optimizer_gen.apply_gradients(zip(gradients_gen, generator.trainable_variables))

    return gen_loss, disc_loss

# miliseconds
print(datetime.now().timetz())
time_list = []
# time in ms
cur_time = int(round(time.time()*1000))
# Run training for the given number of steps.
for step, (batch_x, _) in enumerate(train_data.take(training_steps + 1)):
    if step == 0:
        # Generate noise.
        noise = np.random.normal(-1., 1., size=[batch_size, noise_dim]).astype(np.float32)
        gen_loss = generator_loss(discriminator(generator(noise)))
        disc_loss = discriminator_loss(discriminator(batch_x), discriminator(generator(noise)))
        next_time = int(round(time.time()*1000))
        time_list.append(next_time - cur_time)
        cur_time = next_time
        print("initial: gen_loss: %f, disc_loss: %f" % (gen_loss, disc_loss))
        print('peak active bytes(MB): {}'.format(tf.experimental.get_peak_bytes_active(0)/1024.0/1024.0))
        print('bytes in use(MB): {}'.format(tf.experimental.get_bytes_in_use(0)/1024.0/1024.0))
        continue

    # Run the optimization.
    gen_loss, disc_loss = run_optimization(batch_x)
    next_time = int(round(time.time()*1000))
    time_list.append(next_time - cur_time)
    cur_time = next_time
    if step % display_step == 0:
        print("step: %i, gen_loss: %f, disc_loss: %f" % (step, gen_loss, disc_loss))
        print('peak active bytes(MB): {}'.format(tf.experimental.get_peak_bytes_active(0)/1024.0/1024.0))
        print('bytes in use(MB): {}'.format(tf.experimental.get_bytes_in_use(0)/1024.0/1024.0))

print('throughput: {} ms!!!'.format(np.average(np.array(time_list))))
