import tensorflow as tf
import numpy as np

# tf.data is used only for efficient TFRecord I/O; batches are
# converted to NumPy arrays before being fed to JAX.

_EXAMPLE_FEATURE = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'heatmap': tf.io.FixedLenFeature([], tf.string),
}


def _parse(example):
    features = tf.io.parse_example(example, features=_EXAMPLE_FEATURE)
    image = tf.io.parse_tensor(features['image'], tf.uint8)
    heatmap = tf.io.parse_tensor(features['heatmap'], tf.float32)
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image, heatmap


def create_dataset(tfrecords, batch_size, is_train=False):
    dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def numpy_iter(dataset):
    """Yield (image, heatmap) batches as NumPy arrays for JAX consumption."""
    for image, heatmap in dataset:
        yield np.array(image), np.array(heatmap)
