import jax
import jax.numpy as jnp
import flax.linen as nn


class BottleneckBlock(nn.Module):
    filters: int
    downsample: bool = False

    @nn.compact
    def __call__(self, x, training: bool = False):
        identity = x
        if self.downsample:
            identity = nn.Conv(self.filters, kernel_size=(1, 1), padding='SAME',
                               kernel_init=nn.initializers.he_normal())(x)

        out = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
        out = nn.relu(out)
        out = nn.Conv(self.filters // 2, kernel_size=(1, 1), padding='SAME',
                      kernel_init=nn.initializers.he_normal())(out)

        out = nn.BatchNorm(momentum=0.9)(out, use_running_average=not training)
        out = nn.relu(out)
        out = nn.Conv(self.filters // 2, kernel_size=(3, 3), padding='SAME',
                      kernel_init=nn.initializers.he_normal())(out)

        out = nn.BatchNorm()(out, use_running_average=not training)
        out = nn.relu(out)
        out = nn.Conv(self.filters, kernel_size=(1, 1), padding='SAME',
                      kernel_init=nn.initializers.he_normal())(out)

        return identity + out


class LinearLayer(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Conv(self.filters, kernel_size=(1, 1), padding='SAME',
                    kernel_init=nn.initializers.he_normal())(x)
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
        x = nn.relu(x)
        return x


class Hourglass(nn.Module):
    order: int
    filters: int

    @nn.compact
    def __call__(self, inputs, training: bool = False):
        x = BottleneckBlock(self.filters, name='input_bottleneck')(inputs, training=training)
        up = BottleneckBlock(self.filters, name='up_bottleneck')(x, training=training)

        lower = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        lower = BottleneckBlock(self.filters, name='lower_pre')(lower, training=training)

        if self.order == 1:
            lower = BottleneckBlock(self.filters, name='lower')(lower, training=training)
        else:
            lower = Hourglass(self.order - 1, self.filters, name='lower')(lower, training=training)

        lower = BottleneckBlock(self.filters, name='lower_pre_up_sampling')(lower, training=training)
        # nearest-neighbour 2x upsampling (equivalent to UpSampling2D(size=2))
        lower = jnp.repeat(jnp.repeat(lower, 2, axis=1), 2, axis=2)

        return up + lower
