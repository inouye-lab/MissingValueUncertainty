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

python calibrate_dataset.py starcraft --output ./results/calibration/starcraft/ \
    --dataset '{ "path": "../../datasets/starcraftimage", "image_format": "cifar10", "image_size": 224, "sensor_size": 56 }' \
    --classifier "./models/starcraft-cifar10/starcraft-$model.pklz" \
    --cuda_index $cuda --drop block-dropout \
    --calibration_scales 0.1 0.25 0.5 0.75 1 2.5 5 7.5 10 \
    --beta_variance_scales 0.5 0.99 --zero_imputation \
    --action_spaces '{"name": "binary", "parameter": 0.1}' \
    '{"name": "binary", "parameter": 0.2}' '{"name": "binary", "parameter": 0.3}' \
    '{"name": "binary", "parameter": 0.4}' '{"name": "binary", "parameter": 0.5}' \
    '{"name": "binary", "parameter": 0.6}' '{"name": "binary", "parameter": 0.7}' \
    '{"name": "binary", "parameter": 0.8}' '{"name": "binary", "parameter": 0.9}' \
    --threads 4 --trials 4