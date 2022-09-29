# Low-Resource Machine Translation

<b>!!! This repository is still in an early stage and needs some future-proofing and refactoring first !!!</b>

## Table of Contents

* [Introduction](#introduction)
* [Installation](#installation)
* [Getting started](#getting-started)
* [Results](#results)
  * [Effects of Pivoting in a low-resource setting with (German-Dutch-English) as (source-pivot-target) triplet](#effects-of-pivoting-in-a-low-resource-setting-with-german-dutch-english-as-source-pivot-target-triplet)
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

It is recommended to setup a virtual environment with miniconda (python version 3.10.4)

```
conda create -n mt python=3.10.4
```

and install the requirements like this:

```
pip install -r requirements.txt
```

NOT YET DONE:
The notebooks are designed to work in Google Colab (TODO add link). They work by either:

1. Cloning the repository into your Google Drive root folder and opening a jupyter notebook (mount_drive=True).
1. Just executing a notebook in Google Colab (mount_drive=False). Some operations in the notebooks require a lot of memory and Google Colab might crash if too much data needs to be processed in memory.

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

<details><summary>Results</summary>

The experiments use the WikiMatrix [[4]](#4) dataset.

| Language Pair  | Sentences |
| -------------- | --------- |
| German-Dutch   | 0.5M      |
| Dutch-English  | 0.8M      |
| German-English | 1.6M      |

Several pivoting techniques were employed to test their effects on a simulated low-resource task. The table shows the SacreBLEU scores of the models on the test data.

| Number of Sentences | Baseline | DP       | SWP      | RSWP     |
| ------------------- | -------- | -------- | -------- | -------- |
| unlimited           | 36.525   | 37.082   | 37.052   | 37.367   |
| 10k                 | 1.224    | 19.520   | 20.357   | 22.152   |
| 20k                 | 4.850    | 23.341   | 23.583   | 24.683   |
| 50k                 | 13.301   | 26.278   | 26.301   | 27.387   |
| 100k                | 20.077   | 28.461   | 28.412   | 29.050   |

All models reach a reasonable score with unlimited sentences. The performance of the baseline is very low when only given a few thousand sentences. But the pivoting techniques can amend this problem, due to their pretraining on the (source-pivot) and (pivot-target).

</details>

## Credit

The following table shows sources that influenced the development of this repository.

| Source      | Impact      |
| ----------- | ----------- |
| [[5]](#5) | bigs parts of the transformer class (and all its layers) + beam search + SentencePiece training |
| [[6]](#6) | Top-K and top-p filtering function |

## Contributing

This repository is still in an early and immature state. It would be best if a certain quality standard is established first to make the code future proof.

Here is a list of tasks that would help the progress of this repository and research:

* Perform existing (and future) experiments on different source-pivot-target triplets
* Feedback
* Feature requests
* Bug reporting
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
* Multilingual machine translation [[8]](#8) [[11]](#11)
  * One-to-many
  * Many-to-one
  * Many-to-many
  * English centric, non-english centric
* Zero-shot learning in many-to-many translation [[9]](#9) [[10]](#10)
* Adding speech into the translation directions (zero shooting transcribing, even while translating)
* More elaborate many-to-many architectures [[7]](#7)
* Transfer learning with pivot adapters [[2]](#2)

### Misc

+ Document code!
* Create a central hub for models and tokenizers that were produced with this repo (export format of parameters etc. should be made future proof first)
* Datasets should be easily switchable (download+unzip or use local files) [WikiMatrix is currently hard coded]
* Tokenizers should be trainable with any number of files
* Unit tests (+ automatic execution via GitHub Actions)
* Git Hooks that automatically convert all .ipynb to .py files and create .sh-scripts that process all required arguments for the .py file (pre-receive).
* Option to not save all preprocessed data to disk
* Option to turn use `collate_fn` instead of loading all data into memory
* Better class, method and argument documentation
* More model types and variations to compare transformer performance with other models
* Add more metrics (e.g. ter, translation perplexity)
* Non-autoregressive Transformer (NAT)
* Attention variations of transformers
* Speed-up techniques of transformers
* Training options
  * Soft labels
  * Optimizer and its params
  * Scheduling
  * Dropout rate, etc.
  * Early stopping (saving and using best model after ending training due to metric not improving for n epochs)
* Different types of tokenizers
  * Subword-based (bpe, unigram, word-piece)
  * Word-based
  * Character-based

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
* Multi-GPU training

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
* <a id="8">[8]</a> Multi-Way, Multilingual Neural Machine Translation with a Shared Attention Mechanism (https://arxiv.org/pdf/1601.01073.pdf)
* <a id="9">[9]</a> Zero-Resource Translation with
Multi-Lingual Neural Machine Translation (https://arxiv.org/pdf/1606.04164.pdf)
* <a id="10">[10]</a> Google’s Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation (https://arxiv.org/pdf/1611.04558.pdf)
* <a id="11">[11]</a> MULTI-TASK SEQUENCE TO SEQUENCE LEARNING (https://arxiv.org/pdf/1511.06114.pdf)
