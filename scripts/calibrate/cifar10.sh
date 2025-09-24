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

python calibrate_dataset.py cifar10 --output ./results/calibration/cifar10/ \
    --dataset '{ "path": "../../datasets/cifar10", "image_size": 224, "sensor_size": 56 }' \
    --classifier "./models/cifar10/cifar10-$model.pklz" \
    --cuda_index $cuda --drop block-dropout \
    --calibration_scales 0.1 0.25 0.5 0.75 1 2.5 5 7.5 10 \
    --beta_variance_scales 0.5 --probability_scales 1 --imputators "./models/mean/cifar10.pklz" \
    --action_spaces zero-one \
    --threads 4 --trials 4