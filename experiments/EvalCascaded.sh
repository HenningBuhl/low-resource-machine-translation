#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mt
jupyter nbconvert --to python EvalCascaded.ipynb
python3 EvalCascaded.py
