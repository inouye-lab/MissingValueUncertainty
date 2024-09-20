#!/bin/bash
# Generates the image cache for the CelebA dataset using the CoPaint diffusion model with the bottom mask
source ../../miniconda/bin/activate
conda activate ./venv

python create_generator_cache.py celeba \
    --dataset '{
      "path": "../../datasets/CelebAMask/256/img",
      "lists_root": "datasets/celeba",
      "attributes_path": "../../datasets/CelebAMask/1024/CelebAMask-HQ-attribute-anno.txt",
      "return_index": true
    }' \
    --generator '{
      "name": "gaussian-diffusion",
      "diffusion_batch": 3,
      "model_path": "./models/checkpoints/celeba256_250000.pt"
    }' \
    --samples 30 --cuda_index 1 --seed 1337 --mask bottom --cache_directory ../../datasets/CelebAMask/cache/256/bottom_batches