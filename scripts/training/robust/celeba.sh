#!/bin/bash

# Print usage if unspecified
if [ $# -le 1 ]; then
  echo "Expected arguments: <feature> <cuda>"
  exit 1
fi

feature=$1
cuda=$2
shift
shift
shift
shift

source ../../miniconda/bin/activate
conda activate ./venv

python learn_dirichlet_network.py celeba --seed 1337 --output ./models/robust/celeba/$feature/ \
    --validate_every 10 --training_iterations 1000 --batch_size 250 --patience 5 --evaluate_training \
    --cuda_index $cuda \
    --clean_weight 0 --masked_weight 1 --dirichlet_weight 0 \
    --masks "top" "bottom" "full" "none" \
    --architecture '{"name": "resnet_dirichlet", "activation": "identity", "flatten_single_class": false}' --use_logits \
    '{
      "path": "../../datasets/CelebAMask/256/img",
      "lists_root": "datasets/celeba",
      "targets": ["'$feature'"],
      "attributes_path": "../../datasets/CelebAMask/1024/CelebAMask-HQ-attribute-anno.txt"
    }'
