#!/bin/bash

# Print usage if unspecified
if [ $# -le 1 ]; then
  echo "Expected arguments: <model> <cuda> [sharpness] [calibration]"
  exit 1
fi
model=$1
cuda=$2
sharpness=${3:--10}
calibration=""
zero_var="--zero_variance"
if [ $# -ge 4 ]; then
  calibration="--calibration_map ./results/calibration/mnist/$4.csv"
  zero_var=""
fi
shift $#

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python mvce_dataset.py mnist --output ./results/mnar/mvce/baseline/mnist/ \
    --dataset '{ "path": "../../datasets/mnist", "image_size": 224, "sensor_size": 56 }' \
    --classifier "./models/mnist/mnist-$model.pklz" \
    --cuda_index $cuda --drop '{ "name": "mnar-block-dropout", "sharpness": '$sharpness', "aggregator": "mean" }' \
    $zero_var $calibration --beta_variance_scales 0.5 --probability_scales 10 --imputators "./models/mean/mnist.pklz" \
    --action_spaces zero-one --threads 4 --trials 4