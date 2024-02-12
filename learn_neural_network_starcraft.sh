# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python learn_neural_network.py starcraft --seed 1337 --architecture image_regression --output ./models/starcraft1/ \
    --validate_every 5 --training_iterations 25 --batch_size 100 \
    '{"path": "../../datasets/starcraftimage", "targets": ["player_1_army_count", "player_1_food_workers", "player_2_army_count", "player_2_food_workers", "game_duration_seconds"]}'
