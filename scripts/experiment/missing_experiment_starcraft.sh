#!/bin/bash
# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

name="starcraft"
regressor="starcraft1"
date="20240219-161400"

experiment() {
  local feature="$1"
  local feature_index="$2"
  shift
  shift
  echo
  echo
  echo "Running $name with $feature"
  python missing_experiments.py $name \
    --dataset "{\"path\": \"../../datasets/starcraftimage\", \"targets\": [\"$feature\"], \"samples\": { \"train\": 1000, \"validate\": 1000, \"test\": 1000}}" \
    --regressor "./models/$regressor/$name-$date.pklz" --regressor_feature "$feature_index" --gaussian_path "./datasets/gaussian/$name.pklz" \
    --output "./results/$name/alpha/" --missing 0 0.25 0.5 0.75 1.0 --seed 1337 --residual_batch 100 --threads 8 --gaussian_force_numpy \
     "$@"
}

# feature order from training: "player_1_army_count", "player_1_food_workers", "player_2_army_count", "player_2_food_workers", "game_duration_seconds"

experiment "player_1_army_count" 0 --mc_samples 10  --method_batch 100 --empirical_batch 1000 --empirical_limit 1000
experiment "player_1_army_count" 0 --mc_samples 100 --method_batch 10  --skip_basic_imputation
#experiment "20240219-161400" "player_1_food_workers" 1
#experiment "20240219-161400" "player_2_army_count"   2
#experiment "20240219-161400" "player_2_food_workers" 3
#experiment "20240219-161400" "game_duration_seconds" 4