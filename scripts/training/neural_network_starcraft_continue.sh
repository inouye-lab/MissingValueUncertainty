# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

name=starcraft

learn() {
  local date="$1"
  shift
  echo
  echo
  echo "Learning $name"
  python learn_neural_network.py $name --seed 1234 --input "./models/starcraft1/$name-$date.pklz" --output ./models/starcraft2/ \
    --validate_every 5 --training_iterations 25 --batch_size 100 \
    '{"path": "../../datasets/starcraftimage", "targets": ["player_1_army_count", "player_1_food_workers", "player_2_army_count", "player_2_food_workers", "game_duration_seconds"]}'
}

learn "20240215-173147"