## Reproducing experiments

The `experiments.sh` script in the root directory reproduces quantitative exerpiments from the paper.

 - It uses `sample.py` to create directories of 10,000 examples of synthesized images
   from either an unmodified model, or models that have been edited according to 
   specific user-provided instructions.
 - It uses `seg_stats.py` to compute segmentations of all the unedited and edited images.
 - `seg_correct_mod.py` and `distances.py` are used to calculate correctly modified pixels
    as well LPIPS distances quantifying undesired changes.
