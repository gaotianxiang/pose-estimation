# JAX/Flax Implementation of *Stacked Hourglass Networks for Human Pose Estimation*

This repository is a JAX/Flax implementation of the paper *Stacked Hourglass Networks for Human Pose Estimation* by Newell et al.

## Dataset and Preprocessing

We use the [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). Download and extract the dataset to a local folder.

Run the following command to preprocess the dataset:

```
python preprocess_data.py --path [path to dataset folder]
```

This script parses and preprocesses the dataset and generates TFRecords for fast loading during training.

## Train

To train a model:

```
python train.py --num_stack [number of stacked hourglass modules]
                --resume    [store true, resume from checkpoint if it exists]
                --dtst_path [path to MPII dataset]
```

You can stack any number of hourglass modules via `--num_stack`. Checkpoints are saved to `./ckpt/num_stack_<N>/best.pkl` and the best model (lowest validation loss) is kept automatically.

## Evaluation

To run inference on a single image:

```
python val.py --image_path [path to test image]
              --num_stack  [number of stacked hourglass modules]
              --dtst_path  [path to MPII dataset]
```

To evaluate on 20 randomly sampled images from the dataset, add `--train` or `--test`:

```
python val.py --num_stack [number of stacked hourglass modules]
              --dtst_path [path to MPII dataset]
              --train     [sample from training set]
              --test      [sample from validation set]
```

When `--train` or `--test` is set, `--image_path` is ignored. Skeleton visualizations are saved to `./test/`.

## Requirements

- Python 3.8+
- JAX
- Flax
- Optax
- TensorFlow (for TFRecord I/O only)
- Ray (for parallel TFRecord generation)
- Pillow
- Matplotlib
