#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mt
jupyter nbconvert --to python TrainBaseline.ipynb
python3 TrainBaseline.py
