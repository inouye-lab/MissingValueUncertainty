#!/bin/bash
# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

experiment() {
  local name="$1"
  local date="$2"
  echo
  echo
  echo "Running $name"
  python missing_experiments.py $name --regressor "./models/ridge/$name-$date.pklz" \
    --output ./results/mice4/ --missing 0 0.25 0.5 0.75 1.0 --mc_samples 10 100 1000 \
    --mice_iterations 1 10 100 --seed 1337 --feature_impact --inverted_feature_impact
}

experiment abalone "20231109-003846"
experiment delta_ailerons "20231109-003849"
experiment insurance "20231109-003853"
#experiment elevators "20231109-003851"