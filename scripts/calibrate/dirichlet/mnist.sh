#!/bin/bash

# Print usage if unspecified
if [ $# -le 1 ]; then
  echo "Expected arguments: <model> <cuda> [all]"
  exit 1
fi
model=$1
cuda=$2
all=""
if [ $# -ge 3 ]; then
  all="
    --zero_variance --beta_variance_scales 0.5 0.99 --zero_imputation
  "
  shift
fi
shift $#

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python calibrate_dataset.py mnist --output ./results/calibration/mnist/ \
    --dataset '{ "path": "../../datasets/mnist", "image_size": 224, "sensor_size": 56 }' \
    --classifier "./models/dirichlet-mnist/mnist-$model.pklz" --dmv_classifier \
    --calibration_scales 0.1 0.25 0.5 0.75 1 2.5 5 7.5 10 \
    --cuda_index $cuda --drop block-dropout $all \
    --action_spaces zero-one \
    --threads 4 --trials 4
