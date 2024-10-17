#!/bin/bash
# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python mvce_synthetic.py --output ./results/mvce/ --generator_samples 10 100 1000 --cuda_index -1 \
    --imputation_baselines --mean_shifts 0.25,0.25 0.5,0.5 --flip_variance \
    --correlations -0.7 0 0.6 0.8 0.99 --covariance_scales 0.25 4 \
    --beta_variance_scales 0.5 0.99 \
    --action_spaces zero-one aleatoric --threads 2