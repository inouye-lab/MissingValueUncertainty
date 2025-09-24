#!/bin/bash

cuda=${1:-0}
shift $#

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python learn_mean_imputator.py imagenet '{ "path": "../../datasets/ImageNet" }' \
    --cuda_index $cuda --batch_size 256
