#!/bin/bash
# Learn the gaussian distribution for the starcraft dataset
source ../../miniconda/bin/activate
conda activate ./venv

python learn_gaussian.py starcraft '{"path": "../../datasets/starcraftimage"}' --output ./datasets/gaussian/