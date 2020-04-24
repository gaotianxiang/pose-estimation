import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


class BottleneckBlock(tfkl.Layer):
    def __init__(self, filters, downsample=False):
        super(BottleneckBlock, self).__init__()
        self.filters = filters
        self.down_sample = downsample

    def build(self, input_shape):
        if self.down_sample:
            self.identity_map = tfkl.Conv2D(self.filters, kernel_size=1, padding='same',
                                            kernel_initializer=tfk.initializers.he_normal())

        self.bottleneck = tfk.Sequential([
            tfkl.BatchNormalization(momentum=0.9),
            tfkl.ReLU(),
            tfkl.Conv2D(self.filters // 2, kernel_size=1, padding='same',
                        kernel_initializer=tfk.initializers.he_normal()),

            tfkl.BatchNormalization(momentum=0.9),
            tfkl.ReLU(),
            tfkl.Conv2D(self.filters // 2, kernel_size=3, padding='same',
                        kernel_initializer=tfk.initializers.he_normal()),

            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2D(self.filters, kernel_size=1, padding='same',
                        kernel_initializer=tfk.initializers.he_normal())
        ])

    def call(self, x, **kwargs):
        identity = x
        if self.down_sample:
            identity = self.identity_map(x)
        out = self.bottleneck(x)
        res = identity + out
        return res


class LinearLayer(tfkl.Layer):
    def __init__(self, filters):
        super(LinearLayer, self).__init__()
        self.filters = filters

    def build(self, input_shape):
        self.layer = tfk.Sequential([
            tfkl.Conv2D(self.filters, kernel_size=1, padding='same',
                        kernel_initializer=tfk.initializers.he_normal()),
            tfkl.BatchNormalization(momentum=0.9),
            tfkl.ReLU()
        ])

    def call(self, inputs, **kwargs):
        res = self.layer(inputs)
        return res


class Hourglass(tfkl.Layer):
    def __init__(self, order, filters):
        super(Hourglass, self).__init__()
        self.order = order
        self.filters = filters

    def build(self, input_shape):
        self.input_bottleneck = BottleneckBlock(self.filters)
        self.up_bottleneck = BottleneckBlock(self.filters)
        self.max_pool = tfkl.MaxPool2D(pool_size=2, strides=2)
        self.lower_pre = BottleneckBlock(self.filters)
        self.lower = BottleneckBlock(self.filters) if self.order == 1 else Hourglass(self.order - 1, self.filters)
        self.lower_pre_up_sampling = BottleneckBlock(self.filters)
        self.up_sampling = tfkl.UpSampling2D(size=2)

    def call(self, inputs, **kwargs):
        x = self.input_bottleneck(inputs)
        up = self.up_bottleneck(x)

        lower = self.max_pool(x)
        lower = self.lower_pre(lower)
        lower = self.lower(lower)
        lower = self.lower_pre_up_sampling(lower)
        lower = self.up_sampling(lower)

        res = up + lower
        return res
