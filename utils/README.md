## Sampling methods (used for computing baselines)

`sample.py` is used to sample 50k images (either clean or generated) without edits. Outputs images in form `{img_type}_{imgnum}`, where imgnum is the number passed into `zdataset.z_sample_for_model` as a seed. 


`sample_church.py` is used to get edited images given the original generated images and their imgnums.  Again outputs images form `{img_type}_{imgnum}`, where imgnum is the number passed into `zdataset.z_sample_for_model` as a seed. 


`segmented_samples` generated images which have some percentage of desired pixels from a segmentation class. For example, you can use this to sample churches with domes. Again outputs images form `{img_type}_{imgnum}`, where imgnum is the number passed into `zdataset.z_sample_for_model` as a seed. 

## FID tools, used for supplementary patch graphs and can be used for baseline experiments (though I think Tongzhou probably used a different script)

`get_fid.py` is the main script to precompute several statistics of both cropped images/samples and whole images/samples. It uses `get_samples.py` to get the appropriate samples. The main functions used to obtain cropped statistics are `get_cropped_dataset_statistics` and `get_cropped_fake_statistics`. The main function used to obtain whole image statistics is `get_dataset_statistics`.

## Patch visualization tools 

`visualize_crops.py` lets you visualize the rendering of cropped activations. It is used to produce the patch figures in the supplement. 
