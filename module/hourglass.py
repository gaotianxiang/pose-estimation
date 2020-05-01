import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from module import BottleneckBlock, Hourglass, LinearLayer


class StackedHourglassNetwork(tfk.Model):
    def __init__(self, num_stack=4, num_heatmap=16):
        super(StackedHourglassNetwork, self).__init__()
        self.num_stack = num_stack
        self.num_heatmap = num_heatmap

    def build(self, input_shape):
        self.pre_process = tfk.Sequential([
            tfkl.Conv2D(64, kernel_size=7, strides=2, padding='same',
                        kernel_initializer=tfk.initializers.he_normal()),
            tfkl.BatchNormalization(momentum=0.9),
            tfkl.ReLU(),
            BottleneckBlock(128, downsample=True),
            tfkl.MaxPool2D(pool_size=2, strides=2),
            BottleneckBlock(128, downsample=False),
            BottleneckBlock(256, downsample=True)
        ])

        self.hourglasses = [Hourglass(order=4, filters=256) for _ in range(self.num_stack)]
        self.bottelnecks = [BottleneckBlock(filters=256, downsample=False) for _ in range(self.num_stack)]
        self.linear_layers = [LinearLayer(filters=256) for _ in range(self.num_stack)]
        self.pred_layers = [tfkl.Conv2D(filters=self.num_heatmap, kernel_size=1, padding='same',
                                        kernel_initializer=tfk.initializers.he_normal()) for _ in range(self.num_stack)]
        self.y_restore = [tfkl.Conv2D(filters=256, kernel_size=1, padding='same',
                                      kernel_initializer=tfk.initializers.he_normal())
                          for _ in range(self.num_stack - 1)]
        self.x_post = [tfkl.Conv2D(filters=256, kernel_size=1, padding='same',
                                   kernel_initializer=tfk.initializers.he_normal()) for _ in range(self.num_stack - 1)]

    def call(self, inputs, training=None, mask=None):
        x = self.pre_process(inputs)

        results = []
        for i in range(self.num_stack):
            x = self.hourglasses[i](x)
            x = self.bottelnecks[i](x)
            x = self.linear_layers[i](x)

            pred = self.pred_layers[i](x)
            results.append(pred)

            if i < self.num_stack - 1:
                x_post = self.x_post[i](x)
                pred_restore = self.y_restore[i](pred)
                x = x_post + pred_restore
        return results
