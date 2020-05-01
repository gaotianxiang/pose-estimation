import json
import os
import ray
import tensorflow as tf
import numpy as np
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_train_shards = 64
num_val_shards = 8
ray.init()
tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='/home/tianxiang/dataset/mpii/', type=str)
args = parser.parse_args()

path = args.path


def chunkify(l, n):
    size = len(l) // n
    start = 0
    results = []
    for i in range(n - 1):
        results.append(l[start:start + size])
        start += size
    results.append(l[start:])
    return results


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy(
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def crop_roi(image, keypoint_x, keypoint_y, scale, margin=0.2):
    img_shape = tf.shape(image)
    img_height = img_shape[0]
    img_width = img_shape[1]
    body_height = scale * 200.0

    keypoint_x = tf.cast(keypoint_x, dtype=tf.int32)
    keypoint_y = tf.cast(keypoint_y, dtype=tf.int32)

    # avoid invisible keypoints whose value are -1
    masked_keypoint_x = tf.boolean_mask(keypoint_x, keypoint_x != -1)
    masked_keypoint_y = tf.boolean_mask(keypoint_y, keypoint_y != -1)

    # find \left-most, top, bottom, and right-most keypoints
    keypoint_xmin = tf.reduce_min(masked_keypoint_x)
    keypoint_xmax = tf.reduce_max(masked_keypoint_x)
    keypoint_ymin = tf.reduce_min(masked_keypoint_y)
    keypoint_ymax = tf.reduce_max(masked_keypoint_y)

    # add a padding according to human body height
    xmin = keypoint_xmin - tf.cast(body_height * margin, dtype=tf.int32)
    xmax = keypoint_xmax + tf.cast(body_height * margin, dtype=tf.int32)
    ymin = keypoint_ymin - tf.cast(body_height * margin, dtype=tf.int32)
    ymax = keypoint_ymax + tf.cast(body_height * margin, dtype=tf.int32)

    # make sure the crop is valid
    effective_xmin = xmin if xmin > 0 else 0
    effective_ymin = ymin if ymin > 0 else 0
    effective_xmax = xmax if xmax < img_width else img_width
    effective_ymax = ymax if ymax < img_height else img_height

    image = image[effective_ymin:effective_ymax, effective_xmin:effective_xmax, :]
    new_shape = tf.shape(image)
    new_height = new_shape[0]
    new_width = new_shape[1]

    # shift all keypoints based on the crop area
    effective_keypoint_x = (keypoint_x - effective_xmin) / new_width
    effective_keypoint_y = (keypoint_y - effective_ymin) / new_height

    return image, effective_keypoint_x, effective_keypoint_y


def generate_2d_guassian(height, width, y0, x0, visibility=2, sigma=1, scale=12):
    """
    "The same technique as Tompson et al. is used for supervision. A MeanSquared Error (MSE) loss is
    applied comparing the predicted heatmap to a ground-truth heatmap consisting of a 2D gaussian
    (with standard deviation of 1 px) centered on the keypoint location."
    https://github.com/princeton-vl/pose-hg-train/blob/master/src/util/img.lua#L204
    """
    heatmap = tf.zeros((height, width))

    # this gaussian patch is 7x7, let's get four corners of it first
    xmin = x0 - 3 * sigma
    ymin = y0 - 3 * sigma
    xmax = x0 + 3 * sigma
    ymax = y0 + 3 * sigma
    # if the patch is out of image boundary we simply return nothing according to the source code
    # [1]"In these cases the joint is either truncated or severely occluded, so for
    # supervision a ground truth heatmap of all zeros is provided."
    if xmin >= width or ymin >= height or xmax < 0 or ymax < 0 or visibility == 0:
        return heatmap

    size = 6 * sigma + 1
    x, y = tf.meshgrid(tf.range(0, 6 * sigma + 1, 1), tf.range(0, 6 * sigma + 1, 1), indexing='xy')

    # the center of the gaussian patch should be 1
    center_x = size // 2
    center_y = size // 2

    # generate this 7x7 gaussian patch
    gaussian_patch = tf.cast(tf.math.exp(
        -(tf.square(x - center_x) + tf.math.square(y - center_y)) / (tf.math.square(sigma) * 2)) * scale,
                             dtype=tf.float32)

    # part of the patch could be out of the boundary, so we need to determine the valid range
    # if xmin = -2, it means the 2 left-most columns are invalid, which is max(0, -(-2)) = 2
    patch_xmin = tf.math.maximum(0, -xmin)
    patch_ymin = tf.math.maximum(0, -ymin)
    # if xmin = 59, xmax = 66, but our output is 64x64, then we should discard 2 right-most columns
    # which is min(64, 66) - 59 = 5, and column 6 and 7 are discarded
    patch_xmax = tf.math.minimum(xmax, width) - xmin
    patch_ymax = tf.math.minimum(ymax, height) - ymin

    # also, we need to determine where to put this patch in the whole heatmap
    heatmap_xmin = tf.math.maximum(0, xmin)
    heatmap_ymin = tf.math.maximum(0, ymin)
    heatmap_xmax = tf.math.minimum(xmax, width)
    heatmap_ymax = tf.math.minimum(ymax, height)

    # finally, insert this patch into the heatmap
    indices = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)

    count = 0

    for j in tf.range(patch_ymin, patch_ymax):
        for i in tf.range(patch_xmin, patch_xmax):
            indices = indices.write(count, [heatmap_ymin + j, heatmap_xmin + i])
            updates = updates.write(count, gaussian_patch[j][i])
            count += 1

    heatmap = tf.tensor_scatter_nd_update(heatmap, indices.stack(), updates.stack())

    # unfortunately, the code below doesn't work because
    # tensorflow.python.framework.ops.EagerTensor' object does not support item assignment
    # heatmap[heatmap_ymin:heatmap_ymax, heatmap_xmin:heatmap_xmax] = gaussian_patch[patch_ymin:patch_ymax,patch_xmin:patch_xmax]

    return heatmap


def make_heatmaps(joints_visibility, keypoint_x, keypoint_y, shape=(64, 64, 16)):
    v = tf.cast(joints_visibility, tf.float32)
    x = tf.cast(tf.math.round(keypoint_x * shape[0]), dtype=tf.int32)
    y = tf.cast(tf.math.round(keypoint_y * shape[1]), dtype=tf.int32)

    num_heatmap = shape[2]
    heatmap_array = tf.TensorArray(tf.float32, 16)

    for i in range(num_heatmap):
        gaussian = generate_2d_guassian(shape[1], shape[0], y[i], x[i], v[i])
        heatmap_array = heatmap_array.write(i, gaussian)

    heatmaps = heatmap_array.stack()
    heatmaps = tf.transpose(heatmaps, perm=[1, 2, 0])  # change to (64, 64, 16)

    return heatmaps


def pre_process(annotation, is_train):
    file_path = annotation['filepath']
    image = tf.image.decode_jpeg(tf.io.read_file(file_path))
    keypoint_x = np.array([int(joint[0]) for joint in annotation['joints']])
    keypoint_y = np.array([int(joint[1]) for joint in annotation['joints']])
    visibility = np.array([0 if joint_v == 0 else 1 for joint_v in annotation['joints_visibility']])
    scale = annotation['scale']
    if is_train:
        margin = tf.random.uniform([1], 0.1, 0.3)[0]
    else:
        margin = 0.2
    image, effective_keypoint_x, effective_keypoint_y = crop_roi(image, keypoint_x, keypoint_y, scale, margin)
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.uint8)
    heatmap = make_heatmaps(visibility, effective_keypoint_x, effective_keypoint_y)

    return image, heatmap


def genreate_tfexample(anno, is_train):
    image, heatmap = pre_process(anno, is_train)
    feature = {
        'image': _bytes_feature(tf.io.serialize_tensor(image)),  # tf.uint8
        'heatmap': _bytes_feature(tf.io.serialize_tensor(heatmap))  # tf.float32
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


@ray.remote
def build_single_tfrecord(chunk, path, is_train):
    print('start to build tf records for ' + path)
    with tf.io.TFRecordWriter(path) as writer:
        for anno_list in chunk:
            try:
                tf_example = genreate_tfexample(anno_list, is_train)
            except:
                print(anno_list['filepath'])
            else:
                writer.write(tf_example.SerializeToString())
    print('finished building tf records for ' + path)


def build_tf_records(annotations, total_shards, split):
    chunks = chunkify(annotations, total_shards)
    futures = [
        build_single_tfrecord.remote(
            chunk, os.path.join(path, 'tfr_processed', '{}_{}_of_{}.tfrecords'.format(
                split,
                str(i + 1).zfill(4),
                str(total_shards).zfill(4))),
            split == 'train') for i, chunk in enumerate(chunks)
    ]
    ray.get(futures)


def parse_one_annotation(anno, image_dir):
    filename = anno['image']
    joints = anno['joints']
    joints_visibility = anno['joints_vis']
    annotation = {
        'filename': filename,
        'filepath': os.path.join(image_dir, filename),
        'joints_visibility': joints_visibility,
        'joints': joints,
        'scale': anno['scale'],
        'center': anno['center']
    }
    return annotation


def main():
    print('Start to parse annotations.')
    os.makedirs(os.path.join(path, 'tfr_processed'), exist_ok=True)

    with open(os.path.join(path, 'train.json')) as train_json:
        train_annos = json.load(train_json)
        train_annotations = [
            parse_one_annotation(anno, os.path.join(path, 'images'))
            for anno in train_annos
        ]
        print('First train annotation: ', train_annotations[0])
        del train_annos

    with open(os.path.join(path, 'validation.json')) as val_json:
        val_annos = json.load(val_json)
        val_annotations = [
            parse_one_annotation(anno, os.path.join(path, 'images')) for anno in val_annos
        ]
        print('First val annotation: ', val_annotations[0])
        del val_annos

    print('Start to build TF Records.')
    build_tf_records(train_annotations, num_train_shards, 'train')
    build_tf_records(val_annotations, num_val_shards, 'val')

    print('Successfully wrote {} annotations to TF Records.'.format(len(train_annotations) + len(val_annotations)))


if __name__ == '__main__':
    main()
