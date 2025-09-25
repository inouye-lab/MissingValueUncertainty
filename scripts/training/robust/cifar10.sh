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

python learn_dirichlet_network.py cifar10 --seed 1337 --output ./models/robust/cifar10/ \
    --validate_every 10 --training_iterations 1000 --batch_size 50 --patience 5 \
    --cuda_index $cuda \
    --clean_weight 0 --masked_weight 1 --dirichlet_weight 0 \
    --drop block-dropout \
    --architecture '{"name": "resnet_dirichlet", "activation": "identity"}' --use_logits \
    '{ "path": "../../datasets/cifar10", "image_size": 224, "sensor_size": 56 }'
