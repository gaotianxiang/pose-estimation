import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

from visualize import extract_keypoints_from_heatmap, draw_keypoints_on_image, draw_skeleton_on_image

tfd = tf.data

path = '/home/tianxiang/dataset/mpii/tfr_processed/'

example_feature = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'heatmap': tf.io.FixedLenFeature([], tf.string)
}

dl = tfd.TFRecordDataset([os.path.join(path, fn) for fn in os.listdir(path) if fn.startswith('val')])

for x in dl:
    data = x
    break
data = tf.io.parse_example(data, features=example_feature)
image = tf.io.parse_tensor(data['image'], tf.uint8)
image = tf.cast(image, tf.float32)
image = image / 255
plt.imsave('./img.png', image.numpy())
heatmap = tf.io.parse_tensor(data['heatmap'], tf.float32)
# plt.imshow(image)


kp = extract_keypoints_from_heatmap(heatmap)
draw_skeleton_on_image(image, kp, './skleton.png')
draw_keypoints_on_image(image, kp, save_path='./keypoints.png')
