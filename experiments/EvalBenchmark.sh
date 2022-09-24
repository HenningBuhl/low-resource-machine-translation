#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mt
jupyter nbconvert --to python EvalBenchmark.ipynb
python3 EvalBenchmark.py
