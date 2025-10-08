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

# really just loads then saves it and runs some tests
#    --optimizer '{ "name": "sgd", "lr": 0.0001, "weight_decay": 0.0001, "momentum": 0.9 }' \
python learn_dirichlet_network.py imagenet --seed 1337 --output ./models/dirichlet-imagenet/ \
    --scheduler '{ "name": "cosine-annealing", "T_max": 30, "eta_min": 0 }' \
    --optimizer '{ "name": "sgd", "lr": 0.0001, "weight_decay": 0.0001, "momentum": 0.9 }' \
    --validate_every 5 --training_iterations 30 --batch_size 256 --patience 2 \
    --cuda_index $cuda \
    --clean_weight 1 --masked_weight 1 --dirichlet_weight 0 \
    --drop block-dropout \
    --architecture '{"name": "resnet_dirichlet", "activation": "identity", "keep_final_layer": true}' --use_logits \
    '{ "path": "../../datasets/ImageNet", "sensor_size": 56 }'
