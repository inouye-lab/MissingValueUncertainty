#!/bin/bash

# Print usage if unspecified
if [ $# -le 2 ]; then
  echo "Expected arguments: <feature> <model> <cuda> [prefix]"
  exit 1
fi
feature=$1
model=$2
cuda=$3
prefix=${4:-}
shift $#

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python ece_dataset.py celeba --output ./results/ece/celeba/$feature/ \
    --dataset '{
      "path": "../../datasets/CelebAMask/256/img",
      "lists_root": "datasets/celeba",
      "attributes_path": "../../datasets/CelebAMask/1024/CelebAMask-HQ-attribute-anno.txt",
      "targets": ["'$feature'"]
    }' \
    --classifier "./models/${prefix}celeba/$feature/celeba-$model.pklz" \
    --cuda_index $cuda