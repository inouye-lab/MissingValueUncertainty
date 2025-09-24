#!/bin/bash

# Print usage if unspecified
if [ $# -le 1 ]; then
  echo "Expected arguments: <model> <cuda> [all]"
  exit 1
fi
model=$1
cuda=$2
calibration=""
if [ $# -ge 3 ]; then
  calibration="--calibration_map ./results/calibration/starcraft/$3.csv" # CSV file for dictionary thing
fi
all=""
#if [ $# -ge 3 ]; then
#  all="
#    --zero_variance --beta_variance_scales 0.5 0.99 --zero_imputation
#  "
#  shift
#fi
shift $#

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python mvce_dataset.py starcraft --output ./results/mvce-starcraft-dir/ \
    --dataset '{ "path": "../../datasets/starcraftimage", "image_format": "cifar10", "image_size": 224, "sensor_size": 56 }' \
    --classifier "./models/dirichlet-starcraft-cifar10/starcraft-$model.pklz" --dmv_classifier \
    --cuda_index $cuda --drop block-dropout $all $calibration \
    --action_spaces zero-one --threads 4 --trials 4