import tensorflow as tf

tfd = tf.data

example_feature = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'heatmap': tf.io.FixedLenFeature([], tf.string)
}


def _map(example):
    features = tf.io.parse_example(example, features=example_feature)
    image = tf.io.parse_tensor(features['image'], tf.uint8)
    heatmap = tf.io.parse_tensor(features['heatmap'], tf.float32)
    image = tf.cast(image, tf.float32)
    image = image / 127.5 - 1
    return image, heatmap


def create_dataset(tfrecords, batch_size, is_train=False):
    dataset = tfd.TFRecordDataset(tfrecords, num_parallel_reads=tfd.experimental.AUTOTUNE)
    dataset = dataset.map(_map, num_parallel_calls=tfd.experimental.AUTOTUNE)

    if is_train:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size).prefetch(tfd.experimental.AUTOTUNE)
    return dataset
