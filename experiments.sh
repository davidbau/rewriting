#!/bin/bash
# This script reproduces the quantitative experiments in section
# 5.1 of the paper.  This invokes scripts within metrics/
# to generate large samples of edited images and compute
# metrics evaluating effective changes and undesired changes.

set -ex
# Generate unedited and edited images, 10k each directory.
for CLEAN_EXP in faces church; do
  if [[ ! -e results/samples/${CLEAN_EXP}_clean/done.txt ]]; then
    python -m metrics.sample --dataset ${CLEAN_EXP}
  else
    echo clean images ${CLEAN_EXP} already done
  fi

done

for CLEAN_EXP in faces church; do
  if [[ ! -e results/samples/${CLEAN_EXP}_clean_fid/done.txt ]]; then
    python -m metrics.sample --dataset ${CLEAN_EXP} --fid_samples
  else
    echo clean fid images ${CLEAN_EXP} already done
  fi
done

for EDIT_EXP in smile dome2spire dome2tree dome2castle; do
  if [[ ! -e results/samples/${EDIT_EXP}/done.txt ]]; then
    python -m metrics.sample_edited --mask ${EDIT_EXP}
  else
    echo edited images ${EDIT_EXP} already done
  fi
done

# Get segmentations for buildings
for EXPNAME in church_clean dome2spire faces_clean smile; do
  if [[ ! -e results/samples/seg/${EXPNAME}/done.txt ]]; then
    python -m metrics.seg_stats ${EXPNAME}
  else
    echo segmentations ${EXPNAME} already done
  fi
done

echo 'Running dome2spire'
python -m metrics.seg_correct_mod --exp_name dome2spire
python -m metrics.distances --exp_name dome2spire

echo 'Running smile'
python -m metrics.seg_correct_mod --exp_name smile
python -m metrics.distances --exp_name smile
