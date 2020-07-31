# Rewriting a Deep Generative Model

In this paper, we ask if a deep network can be reprogrammed to follow different rules, by enabling a user to directly change the weights, instead of training with a data set.


<table><tr><td><a href="http://rewriting.csail.mit.edu/"><img src='images/horse-hat-edit.gif'></a><br>
Directly rewriting the weights of a StyleGANv2 to reprogram horses to have hats.</td></tr></table>

### What is model rewriting?
We present the task of *model rewriting*, which aims to add, remove, and alter the semantic and physical rules of a pre-trained deep network.  While modern image editing tools achieve a user-specified goal by manipulating individual input images, we enable a user to synthesize an unbounded number of new images by editing a generative model to carry out modified rules.


### Why rewrite a model?
There are two reasons to want to rewrite a deep network directly:
  1. To gain insight about how a deep network organizes its knowledge.
  2. To enable creative users to quickly make novel models for which there is no existing data set.

Model rewriting envisions a way to construct deep networks according to a user's intentions. Rather than limiting networks to imitating data that we already have, rewriting allows deep networks to model a world that follows new rules that a user wishes to have.


[**Rewriting a Deep Generative Model**](https://rewriting.csail.mit.edu/).<br>
[David Bau](https://people.csail.mit.edu/davidbau/home/), [Steven Liu](http://people.csail.mit.edu/stevenliu/), [Tongzhou Wang](https://ssnl.github.io/), [Jun-Yan Zhu](http://people.csail.mit.edu/junyanz/), [Antonio Torralba](http://web.mit.edu/torralba/www/). <br>
ECCV 2020 (oral).<br>
MIT CSAIL and Adobe Research.

<table><tr><td><center><a href="https://rewriting.csail.mit.edu/paper/"><img height="100" width="78" src="images/paper-thumb.png" style="border:1px solid" data-nothumb=""><br>ECCV 2020<br>Preprint</a></center></td>
<td><center><a href="https://rewriting.csail.mit.edu/video/" class="d-inline-block p-3 align-bottom"><img height="78" width="136" src="images/video-thumb.png" style="border:1px solid" data-nothumb=""><br>ECCV 2020<br>Talk Video</a></center></td>
<td><center><a href="https://rewriting.csail.mit.edu/" class="d-inline-block p-3 align-top"><img height="100" width="78" src="images/website-thumb.png" style="border:1px solid" data-nothumb=""><br>Website</a></center></td>
<td><center><a href="https://colab.research.google.com/github/davidbau/rewriting/blob/master/notebooks/rewriting-interface.ipynb" class="d-inline-block p-3 align-bottom"><img height="78" width="136" src="images/colab-thumb.png" style="border:1px solid" data-nothumb=""><br>Demo Colab<br>Notebook</a></center></td></tr></table>

<img src='images/rewriting_teaser.gif' width="800px" />

Our method rewrites the weights of a generator to change generative rules.
Instead of editing individual images, our method edits the generator, so an infinite set of images can be potentially synthesized and manipulated using the altered rules.  Rules can be changed in various ways, such as *removing* patterns such as watermarks, *adding* objects such as people, or *replacing* definitions such as making trees grow out of towers.

<img src='images/method.png' width="800px" />

Our method is based on the hypothesis that the weights of a generator act as linear associative memory. A layer stores a map between keys, which denote meaningful context, and values, which determine output.

## Example Results

<img src="images/example-eyebrows.png" width=800>
<img src="images/example-tree-towers.png" width=800>
<img src="images/example-horsehats.png" width=800>
<img src="images/example-smiles.png" width=800>
<img src="images/example-erasewindows.png" width=800>

## Tips

The code runs using PyTorch.

* The method and interface can be found in `/rewrite`
* Notebooks are in `/notebooks`: see `rewriting-interface.ipynb` for the demonstration UI.
* Quantitative experiments in `/metrics`, dissection utilities in `/utils`.

## Setup

It's designed to use a recent version of PyTorch (1.4+) on python (3.6), using
cuda 10.1 and cudnn 7.6.0.  The `/setup` directory has a script to create a
conda environment that has the needed dependencies.
