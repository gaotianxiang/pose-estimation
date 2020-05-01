import tensorflow as tf
import tensorflow.keras as tfk
import os
import argparse

from module import StackedHourglassNetwork, loss_func
from data import create_dataset


@tf.function
def train_step(model, loss_func, optimizer, image, heatmap):
    with tf.GradientTape() as tape:
        outs = model(image, training=True)
        loss = sum([loss_func(heatmap, inter_hm) for inter_hm in outs])

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss


@tf.function
def val_step(model, loss_func, image, heapmap):
    outs = model(image, training=False)
    loss = sum([loss_func(heapmap, inter_hm) for inter_hm in outs])
    return loss


def val_epoch(model, loss_func, val_dl):
    loss_mean = tfk.metrics.Mean()
    loss_mean.reset_states()

    for image, heatmap in val_dl:
        loss = val_step(model, loss_func, image, heatmap)
        loss_mean.update_state(loss)

    return loss_mean.result()


def train(model, loss_func, optimizer, train_dl, val_dl, num_epoch, print_interval, ckpt_manager, tfboard_dir):
    train_summary_wwiter = tf.summary.create_file_writer(os.path.join(tfboard_dir, 'trains'))
    test_summary_writer = tf.summary.create_file_writer(os.path.join(tfboard_dir, 'test'))
    best_val_loss = float('inf')
    loss_mean = tfk.metrics.Mean()

    for epoch in range(num_epoch):
        print('start epoch {}'.format(epoch))
        loss_mean.reset_states()
        for step, (image, heatmap) in enumerate(train_dl):
            loss = train_step(model, loss_func, optimizer, image, heatmap)

            loss_mean.update_state(loss)
            if step % print_interval == 0:
                print('\tstep {} mean loss {}'.format(step, loss_mean.result()))
                with train_summary_wwiter.as_default():
                    tf.summary.scalar('loss/steps', loss, step=optimizer.iterations)
        print('epoch {} done train mean loss {}'.format(epoch, loss_mean.result()))
        with train_summary_wwiter.as_default():
            tf.summary.scalar('loss/epoch', loss_mean.result(), step=epoch)

        loss_val = val_epoch(model, loss_func, val_dl)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss/epoch', loss_val, step=epoch)
        print('epoch {} done val mean loss {}'.format(epoch, loss_val))
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            print('ckpt stored')
            ckpt_manager.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_stack', default=1, type=int)
    parser.add_argument('--resume', '--r', action='store_true')
    parser.add_argument('--gpu', default="0", type=str)
    parser.add_argument('--dtst_path', default='/home/tianxiang/dataset/mpii/', type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    num_stack = args.num_stack
    batch_size = 16
    lr = 1e-4
    tfrecords_path = os.path.join(args.dtst_path, 'tfr_processed')
    num_epoch = 100
    ckpt_dir = '/home/tianxiang/codes/pose_estimation_codes/ckpt/num_stack_{}'.format(num_stack)
    tfboard_dir = '/home/tianxiang/codes/pose_estimation_codes/logs/num_stack_{}'.format(num_stack)
    os.makedirs(ckpt_dir, exist_ok=True)
    print_interval = 50

    model = StackedHourglassNetwork(num_stack=num_stack)
    train_dl = create_dataset(
        [os.path.join(tfrecords_path, fn) for fn in os.listdir(tfrecords_path) if fn.startswith('train')],
        batch_size=batch_size, is_train=True)
    test_dl = create_dataset(
        [os.path.join(tfrecords_path, fn) for fn in os.listdir(tfrecords_path) if fn.startswith('val')],
        batch_size=batch_size, is_train=False)

    lr_scheduler = tfk.optimizers.schedules.ExponentialDecay(lr, decay_steps=35000, decay_rate=0.1)
    adam = tfk.optimizers.Adam(lr_scheduler)

    ckpt = tf.train.Checkpoint(model=model, optimizer=adam, dl=train_dl)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
    if args.resume:
        ckpt.restore(manager.latest_checkpoint)

    train(model, loss_func, adam, train_dl, test_dl, num_epoch, print_interval, manager, tfboard_dir)


if __name__ == '__main__':
    main()
