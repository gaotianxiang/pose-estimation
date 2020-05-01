# TF2 Implementation of *Stacked Hourglass Networks for Human Pose Estimation*

This repository is a TF2 implemetation of paper *Stacked Hourglass Networks for Human Pose Estimation* by Newell et al. 

## Dataset and Preprocessing

We are using [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The dataset can be downloaded form the link. Download and extract the dataset to a folder. 

Run the following command from terminal
```
python preprocess_data.py --path [path to dataset folder]
```

This script will parse and preprocess the dataset and generate tf-records for quickly loading in the following training phase.

## Train

To train a model
```
python train.py --num_stack [numbers of stacked hourglass module]
                --gpu [which gpu to use]
                --resume [store true, whether to load checkpoints if exists]
                --dtst_path [path to MPII dataset]
```

You can specify any number of stacked hourglass modules by using `--num_stack` parameters. 

## Evaluation

To evaluate the trained model 
```
python val.py --image_path [path to the test image]
              --num_stack [number of stacked hourglass modules to use]
              --gpu [specify which gpu to use]
              --train [store true]
              --test [store true]
```

if `--train` and `--test` are set, the `--image_path` will be ignored. 20 iamges will be randomly sampled from training or testing set and then be evaluted.

## Requirements
- Python 3.7
- Tensorflow 2.1
