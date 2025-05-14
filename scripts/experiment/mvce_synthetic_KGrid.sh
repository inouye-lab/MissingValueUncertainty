#!/bin/bash
# Loads in all relevant datasets
source ../../miniconda3/bin/activate
conda activate ./venv

python mvce_synthetic_KGrid_test.py --output ./results/mvce/ --generator_samples 100 --cuda_index -1 \
   --singleSample_imputation \
   --correlations 0.6 0.8 --covariance_scales 0.25 4 \
   --beta_variance_scales 0.99 \
   --action_spaces zero-one
   --threads 4 --trials 10