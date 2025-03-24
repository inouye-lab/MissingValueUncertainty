#!/bin/bash

# Print usage if unspecified
if [ $# -le 2 ]; then
  echo "Expected arguments: <split> <cuda> <mask> [batch] [groups] [diffusion_batch]"
  exit 1
fi

# fetch arguments before activating so they are not passed along
split=$1
cuda=$2
mask=$3
batch=''
groups=${5:-10}
diffusionBatch=${6:-3}

# optional split argument
if [ $# -ge 4 ]; then
  index=$4
  shuffled=$split
  batch="\"${split}_list\": \"${shuffled}/split_${index}_of_${groups}.flist\","
  # drop optional arguments
  if [ $# -ge 5 ]; then
    if [ $# -ge 6 ]; then
      shift
    fi
    shift
  fi
  shift
fi
shift
shift
shift

# Generates the image cache for the CelebA dataset using the CoPaint diffusion model with the top mask
source ../../miniconda/bin/activate
conda activate ./venv

python create_generator_cache.py celeba \
    --dataset '{
      "path": "../../datasets/CelebAMask/256/img",
      "lists_root": "datasets/celeba",
      "attributes_path": "../../datasets/CelebAMask/1024/CelebAMask-HQ-attribute-anno.txt",
      '"$batch"'
      "return_index": true
    }' \
    --generator '{
      "name": "gaussian-diffusion",
      "diffusion_batch": '"$diffusionBatch"',
      "model_path": "./models/checkpoints/celeba256_250000.pt"
    }' \
    --split $split --mask $mask \
    --samples 30 --cuda_index $cuda --seed 1337 \
    --cache_directory "../../datasets/CelebAMask/cache/256/${mask}_${split}"