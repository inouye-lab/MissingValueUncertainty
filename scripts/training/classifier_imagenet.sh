#!/bin/bash

cuda=${1:-0}
shift $#

source ../../miniconda/bin/activate
conda activate ./venv

# really just loads then saves it and runs some tests
python learn_neural_network.py imagenet --seed 1337 --output ./models/imagenet/ --cuda_index $cuda \
    --training_iterations 0 --batch_size 256 --loss CrossEntropy \
    --architecture '{"name": "resnet", "activation": "identity", "keep_final_layer": true}' \
    '{ "path": "../../datasets/ImageNet" }'