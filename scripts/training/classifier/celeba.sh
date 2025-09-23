#!/bin/bash

# Print usage if unspecified
if [ $# -le 0 ]; then
  echo "Expected arguments: <feature> [cuda]"
  exit 1
fi

feature=$1
cuda=${2:-0}
if [ $# -ge 2 ]; then
  shift
fi
shift

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python learn_neural_network.py celeba --seed 1337 --output ./models/celeba/$feature/ --cuda_index $cuda \
    --validate_every 10 --training_iterations 1000 --batch_size 250 --patience 5 --loss BCELogits \
    --architecture '{"name": "resnet", "momentum": 0.01, "track_running_stats": false, "activation": "identity"}' \
    '{
      "path": "../../datasets/CelebAMask/256/img",
      "lists_root": "datasets/celeba",
      "targets": ["'$feature'"],
      "attributes_path": "../../datasets/CelebAMask/1024/CelebAMask-HQ-attribute-anno.txt"
    }'
# resnet options:
#  momentum
#  track_running_stats
#  pretrained_weights
#     \
