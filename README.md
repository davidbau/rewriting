# Rewriting a Deep Generative Model

In this paper we ask if a state-of-the-art deep network can be reprogrammed to follow different rules, by enabling a user to directly change the weights, instead of training with a data set.

<table><tr><td><a href="http://rewriting.csail.mit.edu/"><img src='images/horse-hat-edit.gif'></a><br>
Directly rewriting the weights of a StyleGANv2 to reprogram horses to have hats.</td></tr></table>

[**Rewriting a Deep Generative Model**](https://rewriting.csail.mit.edu/).<br>
David Bau, Steven Liu, Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, ECCV 2020 (oral).

<table><tr><td><center><a href="https://rewriting.csail.mit.edu/paper/"><img height="100" width="78" src="images/paper-thumb.png" style="border:1px solid" data-nothumb=""><br>ECCV 2020<br>Preprint</a></center></td>
<td><center><a href="https://rewriting.csail.mit.edu/video/" class="d-inline-block p-3 align-bottom"><img height="78" width="136" src="images/video-thumb.png" style="border:1px solid" data-nothumb=""><br>ECCV 2020<br>Talk Video</a></center></td>
<td><center><a href="https://rewriting.csail.mit.edu/" class="d-inline-block p-3 align-top"><img height="100" width="78" src="images/website-thumb.png" style="border:1px solid" data-nothumb=""><br>Website</a></center></td>
<td><center><a href="https://colab.research.google.com/github/davidbau/rewriting/blob/master/notebooks/rewriting-interface.ipynb" class="d-inline-block p-3 align-bottom"><img height="78" width="136" src="images/colab-thumb.png" style="border:1px solid" data-nothumb=""><br>Demo Colab<br>Notebook</a></center></td></tr></table>

<img src='images/rewriting_teaser.gif' width="800px" />

Our method rewrites the weights of a generator to change generative rules. 
Instead of editing individual images, our method edits the generator, so an infinite set of images can be potentially synthesized and manipulated using the altered rules. 

<img src='images/method.png' width="800px" />

Our method is based on the hypothesis that the weights of a generator act as linear associative memory. A layer stores a map between keys, which denote meaningful context, and values, which determine output. 

## Example Results

<img src="images/example-eyebrows.png">
<img src="images/example-tree-towers.png">
<img src="images/example-horsehats.png">
<img src="images/example-smiles.png">
<img src="images/example-erasewindows.png">

## Tips

The code runs using pytorch.

* The method and interface can be found in `/rewriting`
* Notebooks are in `/notebooks`: see `rewriting-interface.ipynb` for the demonstration UI.
* Quantitative experiments in `/metrics`, dissection utilities in `/utils`.

## Setup

It's designed to use a recent version of pytorch (1.4+) on python (3.6), using
cuda 10.1 and cudnn 7.6.0.  The `/setup` directory has a script to create a
conda environment that has the needed dependencies.

