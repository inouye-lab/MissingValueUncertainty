#!/bin/bash

cuda=$1
calibration=""
zero_var="--zero_variance"
if [ $# -ge 2 ]; then
  calibration="--calibration_map ./results/calibration/synthetic/$2.csv" # CSV file for dictionary thing
  zero_var=""
fi
shift $#

# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

python mvce_synthetic.py --output ./results/mvce/ --generator_samples 100 --cuda_index $cuda --threads 2 \
    $zero_var $calibration --beta_variance_scales 0.99 \
    --single_sample_imputation --mean_imputation \
    --correlations 0.6 0.8 --covariance_scales 0.25 4 \
    --action_spaces zero-one