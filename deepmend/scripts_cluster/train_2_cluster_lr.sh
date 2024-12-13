#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=17
#SBATCH --time=24:00:00

#SBATCH --array=0-0%3

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=michael.schopf@uni-siegen.de
#SBATCH --output=/home/ms152708/DeepMend_changes3_new2/deepmend/scripts_cluster/output%a.txt
#SBATCH --error=/home/ms152708/DeepMend_changes3_new2/deepmend/scripts_cluster/error%a.txt


module load miniconda3
source activate DM





export PYTHONPATH="/home/ms152708/DeepMend_changes3_new2/fracturing"



DATADIR1="/work/ws-tmp/ms152708-DeepMend/test_shapeNet/ShapeNet/ShapeNetCore.v2"
DATADIR2="/work/ws-tmp/ms152708-DeepMend/test_shapeNet_inference2/ShapeNet/ShapeNetCore.v2"
TrainTestFile="/work/ws-tmp/ms152708-DeepMend/test_shapeNet/ShapeNet/ShapeNetCore.v2/split_original/mugs_split.json"



python python/train_2_all_cluster_lr.py \
    --datadir_1 "$DATADIR1" \
    --datadir_2 "$DATADIR2" \
    --train_test_file "$TrainTestFile" \
    --k_obj  1\
    --lr_schedule $SLURM_ARRAY_TASK_ID 500  0.001