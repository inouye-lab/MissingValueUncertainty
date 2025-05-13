#!/bin/bash

feature=$1
shift

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python mvce_dataset.py celeba --output ./results/mvce/top/$feature/ --generator_samples 3 30 \
    --dataset "{
      \"path\": \"../../datasets/CelebAMask/256/img\",
      \"lists_root\": \"datasets/celeba\",
      \"attributes_path\": \"../../datasets/CelebAMask/1024/CelebAMask-HQ-attribute-anno.txt\",
      \"test_list\": \"test_shuffled_1_of_5.flist\",
      \"return_index\": true,
      \"targets\": [\"$feature\"]
    }" \
    --classifier ./models/celeba/celeba-20241010-120712.pklz \
    --classifier_feature $feature \
    --cuda_index 0 \
    --mask top --cache_directory ../../datasets/CelebAMask/cache/256/top_batches \
    --beta_variance_scales 0.5 0.99 --zero_imputation --zero_variance \
    --action_spaces zero-one '{"name":"aleatoric", "constantLoss": 0.4}' --threads 4 --trials 4

#for_feature Mouth_Slightly_Open
# bash scripts/experiment/mvce_celeba_top.sh Receding_Hairline
#bash scripts/experiment/mvce_celeba_top.sh Pale_Skin