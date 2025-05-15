#!/bin/bash

cuda=${1:-0}
shift $#

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python learn_neural_network.py cifar10 --seed 1337 --output ./models/cifar10/ --cuda_index $cuda \
    --optimizer '{"name": "sgd", "lr": 0.1, "weight_decay": 0.0001, "momentum": 0.9 }' \
    --validate_every 10 --training_iterations 1000 --batch_size 50 --patience 5 --loss CrossEntropy \
    --architecture '{"name": "resnet", "activation": "identity"}' \
    '{ "path": "../../datasets/cifar10", "image_size": 224 }'