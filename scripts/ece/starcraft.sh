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

python ece_dataset.py starcraft --output ./results/ece/starcraft/ \
    --dataset '{ "path": "../../datasets/starcraftimage", "image_format": "cifar10", "image_size": 224 }' \
    --classifier "./models/${dirichlet}starcraft-cifar10/starcraft-$model.pklz" \
    --cuda_index $cuda