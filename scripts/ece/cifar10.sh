#!/bin/bash

# Print usage if unspecified
if [ $# -le 1 ]; then
  echo "Expected arguments: <model> <cuda> [dirichlet]"
  exit 1
fi
model=$1
cuda=$2
dirichlet=""
if [ $# -ge 3 ]; then
  dirichlet="dirichlet-"
fi
shift $#

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python ece_dataset.py cifar10 --output ./results/ece/cifar10/ \
    --dataset '{ "path": "../../datasets/cifar10", "image_size": 224 }' \
    --classifier "./models/${dirichlet}cifar10/cifar10-$model.pklz" \
    --cuda_index $cuda
