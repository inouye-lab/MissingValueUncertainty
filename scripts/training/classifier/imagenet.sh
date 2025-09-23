#!/bin/bash

cuda=${1:-0}
iterations=${2:-0}
shift $#

source ../../miniconda/bin/activate
conda activate ./venv

# really just loads then saves it and runs some tests
python learn_neural_network.py imagenet --seed 1337 --output ./models/imagenet/ --cuda_index $cuda \
    --scheduler '{ "name": "cosine-annealing", "T_max": 30, "eta_min": 0 }' \
    --optimizer '{ "name": "sgd", "lr": 0.0001, "weight_decay": 0.0001, "momentum": 0.9 }' \
    --training_iterations $iterations --validate_every 5 --patience 2 --batch_size 256 --loss CrossEntropy \
    --architecture '{"name": "resnet", "activation": "identity", "keep_final_layer": true}' \
    '{ "path": "../../datasets/ImageNet" }'