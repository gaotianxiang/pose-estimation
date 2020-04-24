import tensorflow as tf
import tensorflow.keras as tfk
import argparse
import os
import matplotlib.pyplot as plt

tfd = tf.data

from module import StackedHourglassNetwork
from visualize import extract_keypoints_from_heatmap, draw_skeleton_on_image, draw_keypoints_on_image

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', '--path', type=str)
parser.add_argument('--num_stack', '--ns', default=1, type=int)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
ckpt_path = '/home/tianxiang/codes/pose_estimation_codes/ckpt/num_stack_{}'.format(args.num_stack)
tfr_path = '/home/tianxiang/dataset/mpii/tfr_processed/'

example_feature = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'heatmap': tf.io.FixedLenFeature([], tf.string)
}


def _map(example):
    features = tf.io.parse_example(example, features=example_feature)
    image = tf.io.parse_tensor(features['image'], tf.uint8)
    heatmap = tf.io.parse_tensor(features['heatmap'], tf.float32)
    return image, heatmap


def get_image(image_path):
    encoded = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(encoded)
    # inputs = tf.image.resize(image, (256, 256))
    # inputs = tf.cast(inputs, tf.float32) / 127.5 - 1
    # inputs = tf.expand_dims(inputs, axis=0)
    return image


def predict(model, image, file_name, dir_name):
    # file_name = image_path.split('/')[-1].split('.')[0]
    # dir_name = os.path.dirname(image_path)
    # encoded = tf.io.read_file(image_path)
    # image = tf.image.decode_jpeg(encoded)
    inputs = tf.image.resize(image, (256, 256))
    inputs = tf.cast(inputs, tf.float32) / 127.5 - 1
    inputs = tf.expand_dims(inputs, axis=0)
    outputs = model(inputs, training=False)
    heatmap = tf.squeeze(outputs[-1], axis=0).numpy()
    kp = extract_keypoints_from_heatmap(heatmap)
    draw_skeleton_on_image(image, kp, os.path.join(dir_name, '{}_skeleton.png'.format(file_name)))
    # draw_keypoints_on_image(image, kp, save_path=os.path.join(dir_name, '{}_keypoints.png'.format(file_name)))


def restore(model):
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    # model.build((64, 64, 3))
    # model.load_weights(os.path.join(ckpt_path, 'best_model.h5'))


def main():
    model = StackedHourglassNetwork(args.num_stack)
    restore(model)
    if not args.train and not args.test:
        image_path = args.image_path
        image = get_image(image_path)
        predict(model, image, image_path.split('/')[-1].split('.')[0], os.path.dirname(image_path))
        return

    if args.train:
        tfr = [os.path.join(tfr_path, fn) for fn in os.listdir(tfr_path) if fn.startswith('train')]
    elif args.test:
        tfr = [os.path.join(tfr_path, fn) for fn in os.listdir(tfr_path) if fn.startswith('val')]
    else:
        return

    dtst = tfd.TFRecordDataset(tfr).shuffle(1000).take(20).map(_map)

    for i, (img, hm) in enumerate(dtst):
        kp = extract_keypoints_from_heatmap(hm)
        draw_skeleton_on_image(img, kp, os.path.join('./test/{}_gt.png'.format(i)))
        predict(model, img, '{}'.format(i), './test')


if __name__ == '__main__':
    main()
