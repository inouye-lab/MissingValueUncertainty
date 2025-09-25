#!/bin/bash

# Print usage if unspecified
if [ $# -le 1 ]; then
  echo "Expected arguments: <model> <cuda> [prefix]"
  exit 1
fi
model=$1
cuda=$2
prefix=${3:-}
shift $#

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python ece_dataset.py cifar10 --output ./results/ece/cifar10/ \
    --dataset '{ "path": "../../datasets/cifar10", "image_size": 224 }' \
    --classifier "./models/${prefix}cifar10/cifar10-$model.pklz" \
    --cuda_index $cuda
