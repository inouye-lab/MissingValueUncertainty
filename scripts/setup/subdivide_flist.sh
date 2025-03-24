#!/bin/bash
# Script to split test set into smaller sets to subdivide workload across different GPUs

# Print usage if unspecified
if [ $# -le 3 ]; then
  echo "Expected arguments: <folder> input> <output> <groups> [size]"
  exit 1
fi

folder=$1
input=$2
output=$3
groups=$4
size=${5:-2000}
groupSize=$((size / groups))

# first division doesn't need tail
cat ${folder}/${input}.flist | head -n $groupSize > ${folder}/${output}_1_of_${groups}.flist
for (( i=2; i<=groups; i++)); do
  # rest do tail
  cat ${folder}/${input}.flist | tail -n  +$((groupSize * (i - 1) + 1)) | head -n $groupSize > ${folder}/${output}_${i}_of_${groups}.flist
done
