#!/bin/bash

# Print usage if unspecified
if [ $# -le 1 ]; then
  echo "Expected arguments: <model> <cuda> [sharpness] [aggregator] [activation]"
  exit 1
fi

model=$1
cuda=$2
sharpness=${3:--10}
aggregator=${4:-mean}
activation=${5:-exp}
shift $#

source ../../miniconda/bin/activate
conda activate ./venv

python learn_dirichlet_network.py cifar10 --seed 1337 --output ./models/dirichlet-cifar10/ \
    --validate_every 10 --training_iterations 1000 --batch_size 50 --patience 5 \
    --cuda_index $cuda --teacher ./models/cifar10/cifar10-$model.pklz \
    --clean_weight 0 --masked_weight 0 --dirichlet_weight 1 \
    --drop '{ "name": "mnar-block-dropout", "sharpness": '$sharpness', "aggregator": "'$aggregator'" }' \
    --architecture '{"name": "resnet_dirichlet", "activation": "'$activation'"}' \
    '{ "path": "../../datasets/cifar10", "image_size": 224, "sensor_size": 56 }'
