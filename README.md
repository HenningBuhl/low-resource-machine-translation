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

Techniques that perform well in a low-resource setting are the focus. Pivoting is a method that can amend a low-resource situation. The concept of pivoting was the first milestone to implement:

## Installation

It is recommended to setup a virtual environment with miniconda (python version 3.10.4)

```
conda create -n mt python=3.10.4
```

and install the requirements inside the environment like this:

```
conda activate mt
pip install -r requirements.txt
```

## Getting started

You can either run the python files with your arguments

```
python TrainBaseline.py --src-lang de --tgt-lang nl
```

or use the sh.scripts. Arguments that are internally passed to the python program become PascalCase for the sh-scripts.

```
bash TrainBaseline.sh SrcLang=de TgtLang=nl
```

Put parallel corpus data into experiments/data/{src}-{tgt}/{lang} (order can also be {tgt}-{src}) and monolingula data (optional, only used for tokenizer training) into experiments/data/{lang}. The prepared data for de-nl and nl-en experiments might look like this (note that there is on folder for monolingial english data as it is optional):

* experiments
  * data
    * de-nl
      * de
        * dataset-1.txt
        * dataset-2.txt
      * nl
        * dataset-1.txt
        * dataset-2.txt
    * nl-en
      * nl
        * dataset-3.txt
      * en
        * dataset-3.txt
    * de
      * mono-data-1.txt
      * mono-data-2.txt
    * nl
      * mono-data.txt

The folder `experiments` contains notebook files for each experiment. An experiment will create a folder `experiments/runs/{EXPERIMENT}` where the arguments, results and models will be saved.

Each experiment is present as a notebook to enable quick prototyping and a more neat presentation. A notebook can be converted into a python file with 

```
jupyter nbconvert --to python EXPERIMENT.ipynb
```

The sh-scripts also support additional arguments (these are in snake_notation):

* SKIP_CONVERT: Skips the conversion of the notebook to a python file. This allows the python file to be manually edited and used without it being overwritten.
* CONDA_PATH: The path of conda (uses miniconda default path if unspecified)
* CONDA_ENV: The name of the conda env to use

TODO is the order of experiments important? Only make it possible to train a tokenizer in Trainbaseline?

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

* Make code more flexible to different users' environments
  * .sh-scripts should include option to skip `conda activate` call
  * .sh-scripts should include option to activate a conda environment that was parsed to them in the console
  * .sh-scripts should include option to skip the jupyter conversion (currently a separate .sh-script exists for that purpose)
* Code needs to be revamped to enable easy parametrized execution
  * The jupyter notebooks contain a lot of duplicated code (move it to src folder)
  * `argparse` should be used in the notebooks with default parameters so that parameters can be parsed without editing the notebook or the python file

* Document code!
* Consistent args and kwargs (named and unnamed)



* Convert remaining notebooks
* Git Hooks that automatically convert all .ipynb to .py files and create .sh-scripts that process all required arguments for the .py file (pre-receive).
* Unit tests (+ automatic execution via GitHub Actions)
* Accessibility is important (make notebooks easily executable in google colab!)
* Inference methods should work with batch_size > 1



* Setup conventions (import order, documentation, bool args strtobool and always positively formulated, etc...)
* More model types and variations to compare transformer performance with other models
* Non-autoregressive Transformer (NAT)
* Attention variations of transformers
* Speed-up techniques of transformers
* Multi-GPU training

## Contributing

This repository is still in an early and immature state. It would be best if a certain quality standard is established first to make the code future proof.

Here is a list of tasks that would help the progress of this repository and research:

* Perform existing (and future) experiments on different source-pivot-target triplets
* Feedback
* Feature requests
* Bug reporting
* Any task from the [To-Dos](#to-dos)

If you find an error, a mistake, something does not work or you have an idea for a feature or an improvement - do not hesitate to create a GitHub issue on that topic.

Contributing is done via PRs. See the [CONTRIBUTING](/CONTRIBUTING) file.

## License

This project uses an MIT [LICENSE](/LICENSE).

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
