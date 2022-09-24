#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mt
jupyter nbconvert --to python TrainStepWisePivoting.ipynb
python3 TrainStepWisePivoting.py
