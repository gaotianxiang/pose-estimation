import os
import argparse
import pickle
from typing import Any

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state

from module import StackedHourglassNetwork, loss_func
from data import create_dataset
from data.data_loader import numpy_iter


# ---------------------------------------------------------------------------
# Custom TrainState that also carries BatchNorm running statistics
# ---------------------------------------------------------------------------

class TrainState(train_state.TrainState):
    batch_stats: Any


# ---------------------------------------------------------------------------
# JIT-compiled train / val steps
# ---------------------------------------------------------------------------

@jax.jit
def train_step(state: TrainState, images, heatmaps):
    def loss_fn(params):
        outputs, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            images,
            training=True,
            mutable=['batch_stats'],
        )
        total_loss = sum(loss_func(heatmaps, pred) for pred in outputs)
        return total_loss, updates['batch_stats']

    (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)
    return state, loss


@jax.jit
def val_step(state: TrainState, images, heatmaps):
    outputs = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        images,
        training=False,
    )
    return sum(loss_func(heatmaps, pred) for pred in outputs)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: TrainState, ckpt_dir: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'best.pkl')
    data = {
        'params': jax.device_get(state.params),
        'batch_stats': jax.device_get(state.batch_stats),
        'step': int(state.step),
    }
    with open(ckpt_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'  checkpoint saved to {ckpt_path}')


def load_checkpoint(state: TrainState, ckpt_dir: str) -> TrainState:
    ckpt_path = os.path.join(ckpt_dir, 'best.pkl')
    with open(ckpt_path, 'rb') as f:
        data = pickle.load(f)
    state = state.replace(
        params=data['params'],
        batch_stats=data['batch_stats'],
    )
    print(f'  checkpoint restored from {ckpt_path}')
    return state


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def val_epoch(state, val_dl):
    total_loss = 0.0
    n_batches = 0
    for images, heatmaps in numpy_iter(val_dl):
        total_loss += float(val_step(state, images, heatmaps))
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train(state, train_dl, val_dl, num_epoch, print_interval, ckpt_dir):
    best_val_loss = float('inf')

    for epoch in range(num_epoch):
        print(f'start epoch {epoch}')
        running_loss = 0.0
        step = 0

        for images, heatmaps in numpy_iter(train_dl):
            state, loss = train_step(state, images, heatmaps)
            running_loss += float(loss)
            step += 1

            if step % print_interval == 0:
                print(f'\tstep {step}  mean loss {running_loss / step:.6f}')

        print(f'epoch {epoch} done  train mean loss {running_loss / max(step, 1):.6f}')

        val_loss = val_epoch(state, val_dl)
        print(f'epoch {epoch} done  val mean loss {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(state, ckpt_dir)

    return state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_stack', default=1, type=int)
    parser.add_argument('--resume', '--r', action='store_true')
    parser.add_argument('--dtst_path', default='/home/tianxiang/dataset/mpii/', type=str)
    args = parser.parse_args()

    num_stack = args.num_stack
    batch_size = 16
    lr = 1e-4
    num_epoch = 100
    print_interval = 50
    tfrecords_path = os.path.join(args.dtst_path, 'tfr_processed')
    ckpt_dir = f'./ckpt/num_stack_{num_stack}'

    # Data loaders
    train_dl = create_dataset(
        [os.path.join(tfrecords_path, fn)
         for fn in os.listdir(tfrecords_path) if fn.startswith('train')],
        batch_size=batch_size, is_train=True)
    val_dl = create_dataset(
        [os.path.join(tfrecords_path, fn)
         for fn in os.listdir(tfrecords_path) if fn.startswith('val')],
        batch_size=batch_size, is_train=False)

    # Model initialisation
    model = StackedHourglassNetwork(num_stack=num_stack)
    dummy = jnp.zeros((1, 256, 256, 3), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), dummy, training=False)

    lr_schedule = optax.exponential_decay(
        init_value=lr, transition_steps=35000, decay_rate=0.1)
    tx = optax.adam(lr_schedule)

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        batch_stats=variables['batch_stats'],
    )

    if args.resume:
        state = load_checkpoint(state, ckpt_dir)

    train(state, train_dl, val_dl, num_epoch, print_interval, ckpt_dir)


if __name__ == '__main__':
    main()
