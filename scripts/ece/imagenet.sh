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

python ece_dataset.py imagenet --output ./results/ece/imagenet/ \
    --dataset '{ "path": "../../datasets/ImageNet" }' \
    --classifier "./models/${prefix}imagenet/imagenet-$model.pklz" \
    --cuda_index $cuda