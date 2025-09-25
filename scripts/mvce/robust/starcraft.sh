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
  calibration="--calibration_map ./results/calibration/robust/starcraft/$3.csv" # CSV file for dictionary thing
  zero_var=""
fi
shift $#

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

# using zero imputation as the model auto-handles it
python mvce_dataset.py starcraft --output ./results/mvce/robust/starcraft/ \
    --dataset '{ "path": "../../datasets/starcraft", "image_format": "cifar10", "image_size": 224, "sensor_size": 56 }' \
    --classifier "./models/robust/starcraft-cifar10/starcraft-$model.pklz" \
    --cuda_index $cuda --drop block-dropout \
    $zero_var $calibration --beta_variance_scales 0.5 --probability_scales 10 --zero_imputation \
    --action_spaces zero-one --threads 4 --trials 4