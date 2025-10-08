#!/bin/bash

# Print usage if unspecified
if [ $# -le 0 ]; then
  echo "Expected arguments: <cuda>"
  exit 1
fi

cuda=$1
shift $#

source ../../miniconda/bin/activate
conda activate ./venv

python learn_dirichlet_network.py starcraft --seed 1337 --output ./models/robust/starcraft-cifar10/ \
    --validate_every 10 --training_iterations 1000 --batch_size 250 --patience 5 \
    --cuda_index $cuda \
    --clean_weight 1 --masked_weight 1 --dirichlet_weight 0 \
    --drop block-dropout \
    --architecture '{"name": "resnet_dirichlet", "activation": "identity"}' --use_logits \
    '{ "path": "../../datasets/starcraftimage", "image_format": "cifar10", "image_size": 224, "sensor_size": 56 }'
