#!/bin/bash

OUTPUT="datasets/celeba"
COPAINT_SPLITS="CoPaint/datasets/lama_split"

# CoPaint reshuffles the shuffled train list, so I guess we will to for "consistency"
# that said, we cannot guarantee our new order is the same as their new order
# to make our work reproducible we save the reshuffled order in a directory that won't get deleted
mkdir -p ${OUTPUT}
if [ ! -f ${OUTPUT}/train_val_shuffled.flist ]; then
  echo "Shuffled file does not exist, recreating it. This may lead to inconsistent splits!"
  cat ${COPAINT_SPLITS}/train_shuffled.flist | shuf > ${OUTPUT}/train_val_shuffled.flist
fi
# Split the shuffled file into train and val like [Lama](https://github.com/saic-mdal/lama)
cat ${OUTPUT}/train_val_shuffled.flist | head -n 2000 > ${OUTPUT}/val_shuffled.flist
cat ${OUTPUT}/train_val_shuffled.flist | tail -n +2001 > ${OUTPUT}/train_shuffled.flist
# copy over their visual test as simply test
cat ${COPAINT_SPLITS}/visual_test_shuffled.flist > ${OUTPUT}/test_shuffled.flist