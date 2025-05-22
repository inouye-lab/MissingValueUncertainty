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
all=""
if [ $# -ge 5 ]; then
  all="
    --cache_directory ../../datasets/CelebAMask/cache/256/${mask}_validation
    --generator_samples 3 30
    --zero_variance --beta_variance_scales 0.5 0.99 --zero_imputation --batch_mean_imputation 1
  "
  shift
fi
shift 4

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python calibrate_dataset.py celeba --output ./results/calibration/celeba/$feature/$mask/ \
    --dataset '{
      "path": "../../datasets/CelebAMask/256/img",
      "lists_root": "datasets/celeba",
      "attributes_path": "../../datasets/CelebAMask/1024/CelebAMask-HQ-attribute-anno.txt",
      "return_index": true,
      "targets": ["'$feature'"]
    }' \
    --classifier "./models/dirichlet-celeba/$feature/celeba-$model.pklz" \
    --calibration_scales 0.1 0.25 0.5 0.75 1 2.5 5 7.5 10 \
    --cuda_index $cuda --mask $mask $all \
    --action_spaces '{"name": "binary", "parameter": 0.1}' \
    '{"name": "binary", "parameter": 0.2}' '{"name": "binary", "parameter": 0.3}' \
    '{"name": "binary", "parameter": 0.4}' '{"name": "binary", "parameter": 0.5}' \
    '{"name": "binary", "parameter": 0.6}' '{"name": "binary", "parameter": 0.7}' \
    '{"name": "binary", "parameter": 0.8}' '{"name": "binary", "parameter": 0.9}' \
    --threads 4 --trials 4
