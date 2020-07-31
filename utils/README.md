## Dissection utilities.

David's collection of utilities for performing dissection and
surgery on deep networks in pytorch.

 - nethook simplifies splitting, instrumenting, and modifying
   the behavior of individual steps within a torch.nn.Module.
 - tally and runningstats use the GPU to efficiently compute
   running statistics such as quantile estimations, covariance,
   topk, and counts.
 - labwidget and paintwidget is a small no-dependency UI
   framework for allowing interactive prototyping within a
   notebook, that is compatible both Jupyter and Google Colab.
 - workerpool is a multithreaded utility to facilitate fast
   post-processing and writing of data.
 - segmenter provides a semantic segmenter used for identifying
   semantic concept signals within networks.
 - proggan and stylegan2 are ports of generative networks
   that are compatible with pretrained weights, but that are
   structured for simpler surgery and modification.

