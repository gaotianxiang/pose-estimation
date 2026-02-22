import os
import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

from module import StackedHourglassNetwork
from visualize import extract_keypoints_from_heatmap, draw_skeleton_on_image


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_dir: str):
    ckpt_path = os.path.join(ckpt_dir, 'best.pkl')
    with open(ckpt_path, 'rb') as f:
        data = pickle.load(f)
    return data['params'], data['batch_stats']


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Resize to 256x256, normalise to [-1, 1], add batch dim."""
    img = Image.fromarray(image).resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return arr[None]  # (1, 256, 256, 3)


def predict(model, params, batch_stats, image: np.ndarray, file_name: str, dir_name: str):
    inputs = preprocess_image(image)
    outputs = model.apply(
        {'params': params, 'batch_stats': batch_stats},
        inputs,
        training=False,
    )
    heatmap = np.array(outputs[-1][0])  # (64, 64, 16)
    kp = extract_keypoints_from_heatmap(heatmap)
    os.makedirs(dir_name, exist_ok=True)
    draw_skeleton_on_image(image, kp, os.path.join(dir_name, f'{file_name}_skeleton.png'))


def get_image(image_path: str) -> np.ndarray:
    return np.array(Image.open(image_path).convert('RGB'))


# ---------------------------------------------------------------------------
# TFRecord sampling helpers (reuse tf.data only for I/O)
# ---------------------------------------------------------------------------

_EXAMPLE_FEATURE = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'heatmap': tf.io.FixedLenFeature([], tf.string),
}


def _parse_raw(example):
    features = tf.io.parse_example(example, features=_EXAMPLE_FEATURE)
    image = tf.io.parse_tensor(features['image'], tf.uint8)
    heatmap = tf.io.parse_tensor(features['heatmap'], tf.float32)
    return image, heatmap


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '--path', type=str)
    parser.add_argument('--num_stack', '--ns', default=1, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--dtst_path', default='/home/tianxiang/dataset/mpii/', type=str)
    args = parser.parse_args()

    ckpt_dir = f'./ckpt/num_stack_{args.num_stack}'
    tfr_path = os.path.join(args.dtst_path, 'tfr_processed')

    # Build model and load weights
    model = StackedHourglassNetwork(num_stack=args.num_stack)
    params, batch_stats = load_checkpoint(ckpt_dir)

    if not args.train and not args.test:
        image = get_image(args.image_path)
        file_name = os.path.splitext(os.path.basename(args.image_path))[0]
        predict(model, params, batch_stats, image, file_name, os.path.dirname(args.image_path))
        return

    if args.train:
        tfr = [os.path.join(tfr_path, fn) for fn in os.listdir(tfr_path) if fn.startswith('train')]
    else:
        tfr = [os.path.join(tfr_path, fn) for fn in os.listdir(tfr_path) if fn.startswith('val')]

    os.makedirs('./test', exist_ok=True)
    dataset = tf.data.TFRecordDataset(tfr).shuffle(1000).take(20).map(_parse_raw)

    for i, (img_tf, hm_tf) in enumerate(dataset):
        img = img_tf.numpy()   # uint8 numpy array
        hm = hm_tf.numpy()     # float32 numpy array
        kp_gt = extract_keypoints_from_heatmap(hm)
        draw_skeleton_on_image(img, kp_gt, f'./test/{i}_gt.png')
        predict(model, params, batch_stats, img, str(i), './test')


if __name__ == '__main__':
    main()
