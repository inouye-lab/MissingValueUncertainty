#!/bin/bash
# Loads in all relevant datasets
#source ../../miniconda/bin/activate
#conda activate ./venv

python learn_neural_network.py celeba --seed 1337 --output ./models/celeba-10/ \
    --validate_every 10 --mask '["full", "top", "bottom"]' --training_iterations 1000 --batch_size 250 --patience 5 --evaluate_training \
    --architecture '{"name": "resnet", "momentum": 0.01, "track_running_stats": false}' \
    '{
      "path": "../../datasets/CelebAMask/256/img",
      "lists_root": "datasets/celeba",
      "attributes_path": "../../datasets/CelebAMask/1024/CelebAMask-HQ-attribute-anno.txt"
    }'
# resnet options:
#  momentum
#  track_running_stats
#  pretrained_weights
#     \
