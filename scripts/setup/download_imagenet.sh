#!/bin/bash

cd ../..
mkdir -p datasets/ImageNet
cd datasets/ImageNet

echo "By running this script, you agree to terms of usage for this dataset outlined on https://image-net.org/"

if [ ! -f 'ILSVRC2012_devkit_t12.tar.gz' ]; then
  wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
fi
if [ ! -f 'ILSVRC2012_img_train.tar' ] || [ ! -f 'ILSVRC2012_img_val.tar' ]; then
  echo "Download ImageNet image binaries from https://image-net.org/challenges/LSVRC/2012/2012-downloads.php"
  # ImageNet would prefer I don't distribute the URLs to download the dataset
  # you can easily request access then once you learn the URL add a wget command to the proper URL
  # https://image-net.org/challenges/LSVRC/2012/2012-downloads.php will contain them if you are logged in.
fi
echo "Finished Downloading ImageNet"