#!/bin/bash

# Print usage if unspecified
if [ $# -le 1 ]; then
  echo "Expected arguments: <model> <cuda> <activation>"
  exit 1
fi

model=$1
cuda=$2
activation=$3
shift $#

source ../../miniconda/bin/activate
conda activate ./venv

# really just loads then saves it and runs some tests
#    --optimizer '{ "name": "sgd", "lr": 0.0001, "weight_decay": 0.0001, "momentum": 0.9 }' \
python learn_dirichlet_network.py imagenet --seed 1337 --output ./models/dirichlet-imagenet/ \
    --scheduler '{ "name": "cosine-annealing", "T_max": 30, "eta_min": 0 }' \
    --optimizer '{ "name": "sgd", "lr": 0.0001, "weight_decay": 0.0001, "momentum": 0.9 }' \
    --validate_every 5 --training_iterations 30 --batch_size 256 --patience 2 \
    --cuda_index $cuda --teacher ./models/imagenet/imagenet-$model.pklz \
    --clean_weight 0 --masked_weight 0 --dirichlet_weight 1 \
    --drop block-dropout \
    --architecture '{"name": "resnet_dirichlet", "activation": "'$activation'", "keep_final_layer": true}' \
    '{ "path": "../../datasets/ImageNet", "sensor_size": 56 }'
