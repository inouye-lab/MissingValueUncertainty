#!/bin/bash
# Simple script to unpack the CelebAMask datasets into the format expected

# move into the dataset directory
cd ../../datasets

# start by downloading the two .zip files
# CelebAMask-HQ from https://github.com/switchablenorms/CelebAMask-HQ, mostly for the attributes files
if [ ! -f 'CelebAMask-HQ.zip' ]; then
  echo "Cannot find CelebAMask-HQ.zip, downloading from Google Drive"
  gdown --fuzzy https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view?usp=sharing
fi
if [ ! -f 'CelebAMask-HQ.zip' ]; then
  exit
fi

# Download CelebAMask from https://www.kaggle.com/datasets/kimjiyeop/celebamask-256
if [ ! -f 'CelebAMask-256.zip' ]; then
  echo "Cannot find CelebAMask-256.zip, have not implemented downloading from Kaggle"
  pwd
  echo "Navigate to https://www.kaggle.com/datasets/kimjiyeop/celebamask-256 and place the resulting file in the listed directory as 'CelebAMask-256.zip'"
  exit
fi

# If we made it this far, we are ready to unzip
mkdir -p CelebAMask/256
unzip 'CelebAMask-256.zip' -d CelebAMask/256
# CelebAMask-HQ has an extra directory nested inside, easier to not add that to all our paths
unzip 'CelebAMask-HQ.zip' -d CelebAMask
mv CelebAMask/CelebAMask-HQ CelebAMask/1024

# validate
if [ ! -f 'CelebAMask/1024/CelebAMask-HQ-attribute-anno.txt' ]; then
  echo "Failed to extract CelebAMask-HQ.zip, did not find annotations file"
fi
if [ ! -f 'CelebAMask/256/img/0.jpg' ]; then
  echo "Failed to extract CelebAMask-256.zip, did not find first image"
fi