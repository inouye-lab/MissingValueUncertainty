#!/bin/bash

# Print usage if unspecified
if [ $# -le 3 ]; then
  echo "Expected arguments: <feature> <model> <mask> <cuda> [all]"
  exit 1
fi
feature=$1
model=$2
mask=$3
cuda=$4
calibration=""
if [ $# -ge 5 ]; then
  calibration="--calibration_map ./results/calibration/celeba/$feature/$5.csv" # CSV file for dictionary thing
fi
all=""
if [ $# -ge 6 ]; then
  all="
    --cache_directory ../../datasets/CelebAMask/cache/256/${mask}_test
    --generator_samples 3 30
    --zero_variance --beta_variance_scales 0.5 0.99 --zero_imputation --batch_mean_imputation 1
  "
  shift
fi
shift $#

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python mvce_dataset.py celeba --output ./results/mvce-25/$feature/$mask/ \
    --dataset '{
      "path": "../../datasets/CelebAMask/256/img",
      "lists_root": "datasets/celeba",
      "attributes_path": "../../datasets/CelebAMask/1024/CelebAMask-HQ-attribute-anno.txt",
      "return_index": true,
      "targets": ["'$feature'"]
    }' \
    --classifier "./models/dirichlet-celeba/$feature/celeba-$model.pklz" --dmv_classifier \
    --cuda_index $cuda --mask $mask $all $calibration \
    --action_spaces zero-one --threads 4 --trials 4