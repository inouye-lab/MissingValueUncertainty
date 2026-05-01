#!/bin/bash

# Print usage if unspecified
if [ $# -le 1 ]; then
  echo "Expected arguments: <model> <cuda> [sharpness] [aggregator] [calibration]"
  exit 1
fi
model=$1
cuda=$2
sharpness=${3:--10}
aggregator=${4:-mean}
calibration=""
if [ $# -ge 5 ]; then
  calibration="--calibration_map ./results/calibration/starcraft/$5.csv"
fi
shift $#

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python mvce_dataset.py starcraft --output ./results/mnar/mvce/starcraft/dirichlet/ \
    --dataset '{ "path": "../../datasets/starcraftimage", "image_format": "cifar10", "image_size": 224, "sensor_size": 56 }' \
    --classifier "./models/dirichlet-starcraft-cifar10/starcraft-$model.pklz" --dmv_classifier \
    --cuda_index $cuda --drop '{ "name": "mnar-block-dropout", "sharpness": '$sharpness', "aggregator": "'$aggregator'", "starcraft": true }' \
    --action_spaces zero-one --threads 4 --trials 4 $all $calibration