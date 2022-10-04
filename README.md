# Low-Resource Machine Translation

## Table of Contents

* [Low-Resource Machine Translation](#low-resource-machine-translation)
  * [Table of Contents](#table-of-contents)
  * [Introduction](#introduction)
  * [Installation](#installation)
  * [Getting started](#getting-started)
  * [Experiments](#experiments)
    * [Train a Baseline Model](#train-a-baseline-model)
    * [Direct Pivoting](#direct-pivoting)
    * [Step-wise Pivoting](#step-wise-pivoting)
    * [Reverse Step-wise Pivoting](#reverse-step-wise-pivoting)
    * [Benchmark](#benchmark)
  * [Results](#results)
    * [Effects of Pivoting in a low-resource setting with (German-Dutch-English) as (source-pivot-target) triplet](#effects-of-pivoting-in-a-low-resource-setting-with--german-dutch-english--as--source-pivot-target--triplet)
  * [Credit](#credit)
  * [To-Dos](#to-dos)
  * [Contributing](#contributing)
  * [Conventions](#conventions)
  * [License](#license)
  * [References](#references)

## Introduction

The motivation for this repository was to implement pivoting-related experiments from these papers [[1]](#1) [[2]](#2) [[3]](#3) and make them publicly available.

How to run the implemented experiments is explained in the [experiments](#experiments) section.

## Installation

It is recommended to setup a virtual environment with miniconda (python version 3.10.4)

```
conda create -n mt python=3.10.4
```

and install the requirements inside the environment like this:

```
pip install -r requirements.txt
```

## Getting started

To train a model that translates from one language to another, put parallel corpus data into `experiments/data/{src}-{tgt}/{lang}` (order can also be {tgt}-{src}) and monolingula data (optional, only used for tokenizer training) into `experiments/data/{lang}`. The final folders can contain 1 or more files which must be named consistently across different languages. The prepared data for de-nl and nl-en experiments might look like this (note that there is no folder for monolingial english data as it is optional):

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

You can either run the python files with your arguments

```
python train_one_to_one.py --src-lang de --tgt-lang nl
```

or use the run.sh-script. The script passes arguments to the python program. The script uses PascaslCase for those kind of arguments. Lists are passed comma separated.

```
bash run.sh EXPERIMENT=train_one_to_one SrcLang=de TgtLang=nl
```

The sh-script also support additional arguments (these are in SNAKE_NOTATION) and can be used to run the experiments with qsub or similar programs that require an sh-script.

* EXPERIMENT: The kind of experiment to run (the name of the python file without the .py extension).
* CONDA_PATH: The path of conda (uses miniconda default path if unspecified).
* CONDA_ENV: The name of the conda env to use (it is activated with conda activate {CONDA_ENV}). If not provided, conda will not be used.

## Experiments

Some experiments require preexisting models. When the following sections mention a model being 'saved', it means that a `model.pt` file and an `args.json` file are placed in a directory. This directory can then be used as a `model_path` (or similarly named arguments). It is recommended to save all your models in the `experiments/models` directory for later benchmarking.

### Train a one-to-one Model

This command will train a simple model that translates from the src_lang to the tgt_lang.

```
python train_one_to_one.py --name baseline-de-nl --src-lang de --tgt-lang nl
```

To invesigate the effects of pivoting compared to a baseline model that only uses a very limited amount of data, the argument `--max-examples` can be used to artificially limited the number of sentences.

### Direct Pivoting

You need two models that can perform (src, pvt) and (pvt, tgt) respectively to perform direct pivoting. Save these models and pass their paths to the new training run.

```
python train_one_to_one.py --name dp-de-nl-en --src-lang de --tgt-lang en --encoder-model-path models/baseline-de-nl --decoder-model-path models/baseline-nl-en
```

The `max-examples` is again a good argument to adjust in various runs to test the effects that different amounts of data have on the performance.

### Step-wise Pivoting

To train the step 2 model, use the following command

```
python train_one_to_one.py --name sw-nl-en-step-2 --src-lang nl --tgt-lang en --encoder-model-path models/baseline-de-nl --freeze-encoder True
```

Then perform direct pivoting on the decoder from step 2 and the encoder from the step 1 model.

### Reverse Step-wise Pivoting

Reverse step-wise pivoting is similar to step-wise pivoting but freezes the decoder instead of the encoder. To train the step 2 model, use the following command

```
python train_one_to_one.py --name rsw-de-nl-step-2 --src-lang nl --tgt-lang en --decoder-model-path models/baseline-nl-en --freeze-decoder True
```

Then perform direct pivoting on the encoder from step 2 and the decoder from the step 1 model.

### Benchmark

For benchmarking, save all the models to be benchmarked in the `experiments/models` directory. For cascaded models, only add args.json with content:

```
{
    "model_type": "cascaded",
    "src_lang": "de",
    "pvt_lang": "nl",
    "tgt_lang": "en",
    "src_pvt_model_path": "baseline-de-nl",
    "pvt_tgt_model_path": "baseline-nl-en"
}
```

This will create a cascaded model from the given models. Data for benchmarks is put in `experiments/benchmark`. Each benchmark gets its own folder containing its data. It might look like this:

* experiments
  * benchmark
    * flores
      * de
        * file-1.txt
        * file-2.txt
      * nl
        * file-1.txt
        * file-2.txt
      * en
        * file-1.txt
        * file-2.txt
    * tatoeba
      * de-nl
        * de
          * file.txt
        * nl
          * file.txt
      * de-en
        * de
          * file.txt
        * en
          * file.txt
      * nl-en
        * nl
          * file.txt
        * en
          * file.txt

The final directories can contain 1 or more files, just like the `experiments/data/{src}-{tgt}/{lang}` directories. Some benchmarks provide a set of sentences which have been translated into all available languages (e.g. flores). Others contain parallel data for each src-tgt pair (e.g. tatoeba). This will lead to a different folder setup depending on which kind of benchmark you use. Both types are supported, as seen above.

Now you are ready to benchmark your models on your benchmarks.

```
python eval_benchmark.py
```

## Results

### Effects of Pivoting in a low-resource setting with (German-Dutch-English) as (source-pivot-target) triplet

<details><summary>Results</summary>

The experiments used the WikiMatrix [[4]](#4) dataset.

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

All models reach a reasonable score with unlimited sentences (DE-EN). The performance of the baseline is very low when only given a few thousand sentences. But the pivoting techniques can amend this problem, due to their pretraining on the (source-pivot) and (pivot-target) data. The improvement is especially pronounced in the 10k and 20k sentence cases.

</details>

## Credit

The following table shows sources that influenced the development of this repository.

| Source      | Impact      |
| ----------- | ----------- |
| [[5]](#5) | bigs parts of the transformer class (and all its layers) + beam search + SentencePiece training |
| [[6]](#6) | Top-K and top-p filtering function |

## To-Dos

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
* Non-autoregressive Transformer (NAT)
* Attention variations of transformers
* Speed-up techniques of transformers

## Contributing

This repository is still in an early and immature state. It would be best if a certain quality standard is established first to make the code future proof.

Here is a list of tasks that would help the progress of this repository and research:

* Use it to experiment!
* Perform existing (and future) experiments on different source-pivot-target triplets
* Feedback
* Feature requests
* Bug reporting

If you find an error, a mistake, something does not work or you have an idea for a feature or an improvement - do not hesitate to create a GitHub issue or a PR on that topic.

Contributing is done via PRs. See the [CONTRIBUTING](/CONTRIBUTING) file.

## Conventions

Conventions can be seen in the [CONVENTIONS](/CONVENTIONS) file.

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
