#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mt
jupyter nbconvert --to python TrainDirectPivoting.ipynb
python3 TrainDirectPivoting.py
