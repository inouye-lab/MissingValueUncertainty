#!/bin/bash
source ../../miniconda/bin/activate
conda activate ./venv

python learn_dirichlet_network.py celeba --seed 1337 --output ./models/dirichlet-celeba/ \
    --validate_every 10 --training_iterations 1000 --batch_size 250 --patience 5 --evaluate_training \
    --cuda_index 1 \
    --masks "top" "bottom" "full" "none" \
    --architecture '{"name": "resnet_dirichlet", "momentum": 0.01, "track_running_stats": false}' \
    '{
      "path": "../../datasets/CelebAMask/256/img",
      "lists_root": "datasets/celeba",
      "targets": ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Bald", "Wearing_Hat"],
      "attributes_path": "../../datasets/CelebAMask/1024/CelebAMask-HQ-attribute-anno.txt"
    }'
# resnet options:
#  momentum
#  track_running_stats
#  pretrained_weights
#     \
