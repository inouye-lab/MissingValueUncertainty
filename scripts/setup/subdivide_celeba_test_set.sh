#!/bin/bash
# Script to split test set into smaller sets to subdivide workload across different GPUs

OUTPUT="datasets/celeba"

# original one is sampling from 0 to 799
# lines 800 onwards is the first split
cat ${OUTPUT}/test_shuffled.flist | head -n 400 > ${OUTPUT}/test_shuffled_1_of_5.flist
cat ${OUTPUT}/test_shuffled.flist | tail -n  +401 | head -n 400 > ${OUTPUT}/test_shuffled_2_of_5.flist
cat ${OUTPUT}/test_shuffled.flist | tail -n  +801 | head -n 400 > ${OUTPUT}/test_shuffled_3_of_5.flist
cat ${OUTPUT}/test_shuffled.flist | tail -n +1201 | head -n 400 > ${OUTPUT}/test_shuffled_4_of_5.flist
cat ${OUTPUT}/test_shuffled.flist | tail -n +1601 | head -n 400 > ${OUTPUT}/test_shuffled_5_of_5.flist
