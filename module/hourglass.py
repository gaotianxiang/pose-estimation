import jax.numpy as jnp
import flax.linen as nn

from module.layers import BottleneckBlock, Hourglass, LinearLayer


class StackedHourglassNetwork(nn.Module):
    num_stack: int = 4
    num_heatmap: int = 16

    def setup(self):
        self.pre_conv = nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='SAME',
                                kernel_init=nn.initializers.he_normal())
        self.pre_bn = nn.BatchNorm(momentum=0.9)
        self.pre_bottleneck_1 = BottleneckBlock(128, downsample=True)
        self.pre_bottleneck_2 = BottleneckBlock(128, downsample=False)
        self.pre_bottleneck_3 = BottleneckBlock(256, downsample=True)

        self.hourglasses = [Hourglass(order=4, filters=256) for _ in range(self.num_stack)]
        self.bottlenecks = [BottleneckBlock(filters=256, downsample=False) for _ in range(self.num_stack)]
        self.linear_layers = [LinearLayer(filters=256) for _ in range(self.num_stack)]
        self.pred_layers = [
            nn.Conv(self.num_heatmap, kernel_size=(1, 1), padding='SAME',
                    kernel_init=nn.initializers.he_normal())
            for _ in range(self.num_stack)
        ]
        if self.num_stack > 1:
            self.y_restore = [
                nn.Conv(256, kernel_size=(1, 1), padding='SAME',
                        kernel_init=nn.initializers.he_normal())
                for _ in range(self.num_stack - 1)
            ]
            self.x_post = [
                nn.Conv(256, kernel_size=(1, 1), padding='SAME',
                        kernel_init=nn.initializers.he_normal())
                for _ in range(self.num_stack - 1)
            ]

    def __call__(self, inputs, training: bool = False):
        # Preprocessing: 256x256 -> 64x64, 3 -> 256 channels
        x = self.pre_conv(inputs)
        x = self.pre_bn(x, use_running_average=not training)
        x = nn.relu(x)
        x = self.pre_bottleneck_1(x, training=training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        x = self.pre_bottleneck_2(x, training=training)
        x = self.pre_bottleneck_3(x, training=training)

        results = []
        for i in range(self.num_stack):
            x = self.hourglasses[i](x, training=training)
            x = self.bottlenecks[i](x, training=training)
            x = self.linear_layers[i](x, training=training)

            pred = self.pred_layers[i](x)
            results.append(pred)

            if i < self.num_stack - 1:
                x = self.x_post[i](x) + self.y_restore[i](pred)

        return results
