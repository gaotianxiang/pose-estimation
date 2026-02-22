import jax.numpy as jnp


def loss_func(labels, output):
    weights = (labels > 0).astype(jnp.float32) * 81 + 1
    return jnp.mean(jnp.square(labels - output) * weights)
