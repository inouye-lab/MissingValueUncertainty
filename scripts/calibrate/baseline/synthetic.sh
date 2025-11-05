#!/bin/bash
# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python calibrate_synthetic.py --output ./results/calibration/synthetic/ --generator_samples 100 --cuda_index -1 \
   --beta_variance_scales 0.99 \
   --single_sample_imputation --mean_imputation \
   --correlations 0.6 0.8 --covariance_scales 0.25 4 \
   --action_spaces zero-one \
   --threads 4 --trials 10