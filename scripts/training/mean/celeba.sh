#!/bin/bash

cuda=${1:-0}
shift $#

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python learn_mean_imputator.py celeba \
    --cuda_index $cuda --batch_size 250 \
    '{
      "path": "../../datasets/CelebAMask/256/img",
      "lists_root": "datasets/celeba",
      "attributes_path": "../../datasets/CelebAMask/1024/CelebAMask-HQ-attribute-anno.txt"
    }'
