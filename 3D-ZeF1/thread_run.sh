#!/bin/bash
source /home/data/anaconda3/etc/profile.d/conda.sh
conda activate FairMOT
conda info --envs

cd /home/huangjinze/code/3D-ZeF
python threading_detect.py

