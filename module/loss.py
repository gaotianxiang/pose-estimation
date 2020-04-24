import tensorflow as tf


def loss_func(labels, output):
    weights = tf.cast(labels > 0, dtype=tf.float32) * 81 + 1
    # loss = tf.reduce_mean(tf.math.square(labels - output))
    loss = tf.math.reduce_mean(tf.math.square(labels - output) * weights)
    return loss
