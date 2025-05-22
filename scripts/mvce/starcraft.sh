#!/bin/bash

# Print usage if unspecified
if [ $# -le 1 ]; then
  echo "Expected arguments: <model> <cuda>"
  exit 1
fi
model=$1
cuda=$2
shift $#

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python mvce_dataset.py starcraft --output ./results/mvce-starcraft/ \
    --dataset '{ "path": "../../datasets/starcraftimage", "image_format": "cifar10", "image_size": 224, "sensor_size": 56 }' \
    --classifier "./models/starcraft-cifar10/starcraft-$model.pklz" \
    --cuda_index $cuda --drop block-dropout \
    --zero_variance --beta_variance_scales 0.5 0.99 --zero_imputation \
    --action_spaces zero-one --threads 4 --trials 4