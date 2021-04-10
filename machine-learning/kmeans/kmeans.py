import numpy as np
import tensorflow as tf
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
parser.add_argument('height_width', type=int,  help="dataset scale, e.g., 32")
parser.add_argument('num_clusters', type=int,  help="dataset scale, e.g., 32")
parser.add_argument('steps', type=int,  help="training steps, e.g., 10")
parser.set_defaults(lms=False)
args = parser.parse_args()

if args.lms:
    tf.config.experimental.set_lms_enabled(True)
    tf.experimental.get_peak_bytes_active(0)

img_h, img_w, channel = args.height_width, args.height_width, 3

num_points = 1024
dimensions = img_h*img_w*channel
points = np.random.uniform(0, 255, [num_points, dimensions])
num_clusters = args.num_clusters
print('==============> shape: {}'.format(points.shape))

def input_fn():
  return tf.compat.v1.train.limit_epochs(
          tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)

# FIXME: TF 2.0 for tf.compat.v1
#kmeans = tf.estimator.experimental.KMeans(
#        num_clusters=num_clusters, use_mini_batch=False)
kmeans = tf.compat.v1.estimator.experimental.KMeans(
        num_clusters=num_clusters, use_mini_batch=False, distance_metric=tf.compat.v1.estimator.experimental.KMeans.COSINE_DISTANCE)

# train
#num_iterations = 5
num_iterations = args.steps
#previous_centers = None
# miliseconds
print(datetime.now().timetz())
time_list = []
# time in ms
cur_time = int(round(time.time()*1000))
for _ in range(num_iterations):
  kmeans.train(input_fn)
  next_time = int(round(time.time()*1000))
  time_list.append(next_time - cur_time)
  cur_time = next_time
  #cluster_centers = kmeans.cluster_centers()
  #if previous_centers is not None:
  #  print('delta:', cluster_centers - previous_centers)
  #previous_centers = cluster_centers
  #print('score:', kmeans.score(input_fn))
  print('==> peak active bytes(MB): {}'.format(tf.experimental.get_peak_bytes_active(0)/1024.0/1024.0))
  print('==> bytes in use(MB): {}'.format(tf.experimental.get_bytes_in_use(0)/1024.0/1024.0))

cluster_centers = kmeans.cluster_centers()
print('cluster centers:', cluster_centers)
print('throughput: {} ms!!!'.format(np.average(np.array(time_list))))

'''
# map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(points):
  cluster_index = cluster_indices[i]
  center = cluster_centers[cluster_index]
  print('point:', point, 'is in cluster', cluster_index, 'centered at', center)
'''
