# Low-Resource Machine Translation

## Table of Contents

* [Introduction](#introduction)
* [Installation](#installation)
* [Getting started](#getting-started)
* [Results](#results)
  * [Effects of Pivoting in a low-resource setting with (German-Dutch-English) as (source-pivot-target) triplet](#effects-of-pivoting-in-a-low-resource-setting-with--german-dutch-english--as--source-pivot-target--triplet)
* [Credit](#credit)
* [Contributing](#contributing)
* [To-Dos](#to-dos)
  * [Experiments](#experiments)
  * [Misc](#misc)
  * [Quality of Life](#quality-of-life)
* [License](#license)
* [References](#references)

## Introduction

The motivation for this repository was to make experiments from these papers [[1]](#1) [[2]](#2) [[3]](#3) publicly available.

Techniques that perform well in a low-resource setting are the focus. Pivoting is a method that can amend a low-resource situation. The concept of pivoting was the first milestone to implement, so the following experiments are available:

* Training a baseline model
* Evaluating a cascaded model
* Training a direct pivoting model (DP)
* Training a step-wise pivoting model (SWP)
* Training a reverse-step-wise pivoting model (RSWP) [similar to step-wise pivoting, but the decoder is pre-trained and gets frozen in step 2 instead of the encoder]
* Evaluating models on benchmarks

## Installation

It is recommended to setup a virtual environment with python version 3.10.4

```
conda create -n mt python=3.10.4
```

and install the requirements like this:

```
pip install -r requirements.txt
```

## Getting started

The folder `experiments` contains notebook files for each experiment. An experiment will create a folder `experiments/runs/{EXPERIMENT}` where hyper parameters, results and models will be saved.

Each experiment is present as a notebook to enable quick prototyping and a more neat presentation. A notebook can be converted into a python file with 

```
jupyter nbconvert --to python EXPERIMENT.ipynb
```

This is what happens in the .sh-scripts. These scripts were made to enable quick experiment execution with qsub on a remote machine or cluster.

The usual workflow for running an experiment is to adjust the hyper parameters in the jupyter notebook and then running it either directly, as a python file after converting it or passing it to a cluster with qsub.

## Results

### Effects of Pivoting in a low-resource setting with (German-Dutch-English) as (source-pivot-target) triplet

## Credit

The following table shows sources that influenced the development of this repository.

| Source      | Impact      |
| ----------- | ----------- |
| [[5]](#5) | bigs parts of the transformer class (and all its layers) + beam search + SentencePiece training |
| [[6]](#6) | Top-K and top-p filtering function |

## Contributing

Here is a list of tasks that would help the progress of this repository and research:

* Perform existing (and future) experiments on different source-pivot-target triplets
* Any task from the [To-Dos](#to-dos)

If you find an error, a mistake, something does not work or you have an idea for a feature or an improvement - do not hesitate to create a GitHub issue on that topic.

## To-Dos

### Experiments

* Cross lingual encoder [[2]](#2)
* Use encoders of different but similar languages (e.g. German-X encoder for Dutch-X translation) either frozen or unfrozen and observe the effects
* Teacher-Student training
* Freeze only the encoder or decoder in direct pivoting to reduce the number of trained parameters (mitigate overfitting)
* Intermediate representation (multi-lingual) in many-to-many translation [[1]](#1)
* Phrase table injection [[1]](#1)
* Zero-shot learning in many-to-many translation
* Adding speech into the translation directions
* More elaborate many-to-many architectures [[7]](#7)
* Transfer learning with pivot adapters [[2]](#2)

### Misc

* Create a central hub for models and tokenizers that were produced with this repo (export format of parameters etc. should be made future proof first)
* Datasets should be easily switchable (download+unzip or use local files) [WikiMatrix is currently hard coded]
* Tokenizers should be trainable with any number of files
* Automated tests
* Option to not save all preprocessed data to disk
* Option to turn use `collate_fn` instead of loading all data into memory
* Better class, method and argument documentation

### Quality of Life

* Make code more flexible to different users' environments
  * .sh-scripts should include option to skip `conda activate` call
  * .sh-scripts should include option to activate a conda environment that was parsed to them in the console
  * .sh-scripts should include option to skip the jupyter conversion (currently a separate .sh-script exists for that purpose)
* Code needs to be revamped to enable easy parametrized execution
  * The jupyter notebooks contain a lot of duplicated code
  * `argparse` should be used in the notebooks with default parameters so that parameters can be parsed without editing the notebook or the python file
* All settings, parameters and results should be exported in a future proof, clean and unified format
* Inference methods should work with batch_size > 1

## License

This project uses an [MIT license](/LICENSE)

## References

* <a id="1">[1]</a> Neural Machine Translation in Low-Resource Setting: a Case Study in English-Marathi Pair (https://www.cse.iitb.ac.in/~pb/papers/mts21-e-m-nmt.pdf)
* <a id="2">[2]</a> Pivot-based transfer learning for neural machine translation between non-English languages (https://arxiv.org/pdf/1909.09524v1.pdf)
* <a id="3">[3]</a> Pivot based transfer learning for neural machine translation (https://aclanthology.org/2021.wmt-1.39.pdf)
* <a id="4">[4]</a> WikiMatrix dataset (https://opus.nlpl.eu/WikiMatrix-v1.php)
* <a id="5">[5]</a> Pytorch Transformer Machine Translation (https://github.com/devjwsong/transformer-translator-pytorch)
* <a id="6">[6]</a> Top-K and Top-p filtering (https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317)
* <a id="7">[7]</a> Beyond English-Centric Multilingual Machine Translation (https://arxiv.org/pdf/1909.09524v1.pdf)
