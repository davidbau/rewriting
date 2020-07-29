## Supplementary patch graphs

`fid_image_patches.py` generates rendered cropped activations, computes their FID, and saves the results to `patch_fid.txt`. This assumes that you have precomputed statistics of cropped samples from `utils/get_fid.py`. 

After `patch_fid.txt` is populated, you can use `plot_patch_fids.py` to plot the results. 

`fid.py` is a standard script to compute FID. It contains some utils to compute FID from `.npz` image batches, `.npz` precomputed image statistics, and pytorch images.  Refer to `fid.sh` for example usage. 

## Metrics for rebuttal/draft:

First, clone the LPIPS repo.

This requires that you have samples to clean images and edited images w/ the image numbers correctly corresponding (for example, for clean image 1 named `clean/clean1.png`, the edited name should be `edit_name/edit_name1.png`). You can refer to `../utils/sample_edited.py` and `../utils/sample.py` for reference. 

Then, generate the segmentation stats by running `seg_stats_aio.py`, with reference script `segment.sh`. Then, we can find the correctly modified pixels with `seg_stats_ait.py`. Be sure that the segmentation class index of source pixel change (for example dome) is properly set in `srcc` (which channel of segmentation map to look) and `src` (segmentation index) fields. Similarly for target class indices for `tgt` and `tgtc`. 

To get the LPIPS numbers, you can refer to `lpips.sh`. To get masked LPIPS numbers, you can look at `unnecessary.sh`. Similarly make sure the segmentation classes are set correctly.
