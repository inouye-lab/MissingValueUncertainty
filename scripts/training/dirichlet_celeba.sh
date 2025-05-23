#!/bin/bash

# Print usage if unspecified
if [ $# -le 1 ]; then
  echo "Expected arguments: <feature> <model> [cuda]"
  exit 1
fi

feature=$1
model=$2
cuda=$3
activation=$4
shift
shift
shift
shift

source ../../miniconda/bin/activate
conda activate ./venv

python learn_dirichlet_network.py celeba --seed 1337 --output ./models/dirichlet-celeba/$feature/ \
    --validate_every 10 --training_iterations 1000 --batch_size 50 --patience 5 --evaluate_training \
    --cuda_index $cuda --teacher ./models/celeba/$feature/celeba-$model.pklz \
    --clean_weight 0 --masked_weight 0 --dirichlet_weight 1 \
    --masks "top" "bottom" "full" "none" \
    --architecture '{"name": "resnet_dirichlet", "activation": "'$activation'"}' \
    '{
      "path": "../../datasets/CelebAMask/256/img",
      "lists_root": "datasets/celeba",
      "targets": ["'$feature'"],
      "attributes_path": "../../datasets/CelebAMask/1024/CelebAMask-HQ-attribute-anno.txt"
    }'
