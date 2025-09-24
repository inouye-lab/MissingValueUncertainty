#!/bin/bash

# Print usage if unspecified
if [ $# -le 1 ]; then
  echo "Expected arguments: <model> <cuda> [calibration]"
  exit 1
fi
model=$1
cuda=$2
calibration=""
zero_var="--zero_variance"
if [ $# -ge 3 ]; then
  calibration="--calibration_map ./results/calibration/mnist/$3.csv" # CSV file for dictionary thing
  zero_var=""
fi
shift $#

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python mvce_dataset.py mnist --output ./results/mvce-mnist/ \
    --dataset '{ "path": "../../datasets/mnist", "image_size": 224, "sensor_size": 56 }' \
    --classifier "./models/mnist/mnist-$model.pklz" \
    --cuda_index $cuda --drop block-dropout \
    $zero_var $calibration --beta_variance_scales 0.5 0.99 --imputators "./models/mean/mnist.pklz" \
    --action_spaces zero-one --threads 4 --trials 4