#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=michael.schopf@uni-siegen.de



module load miniconda3
source activate DM



export DATADIR="/home/ms152708/Data"


export PYTHONPATH="/home/ms152708/DeepMend_ablation/fracturing"





export PYTHONPATH="/home/ms152708/DeepMend_ablation/fracturing"; /home/ms152708/.conda/envs/DM/bin/python python/train_1.py -e /home/ms152708/DeepMend_ablation/deepmend/experiments_infer/bottles_50 -c 4800