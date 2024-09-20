#!/bin/bash
# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

learn() {
  local name="$1"
  shift
  echo
  echo
  echo "Learning $name"
  python learn_neural_network.py $name --output ./models/nn/ --seed 1337 "$@"
}

# input size: 10
learn abalone --layers 16 8 4 --training_iterations 1000
# input size: 5
learn delta_ailerons --layers 8 4 --training_iterations 1000
# input size: 18
learn elevators --layers 24 12 6 --training_iterations 1000 --patience 5
# input size: 11
learn insurance --layers 16 8 4 --training_iterations 2000