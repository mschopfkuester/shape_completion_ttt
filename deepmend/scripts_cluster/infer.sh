#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=15
#SBATCH --time=15:00:00

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=michael.schopf@uni-siegen.de
#SBATCH --output=/work/ws-tmp/ms152708-DeepMend/DeepMend_changes3_new2/output_files_cluster/cars/infer/output%a.txt
#SBATCH --error=/work/ws-tmp/ms152708-DeepMend/DeepMend_changes3_new2/output_files_cluster/cars/infer/error%a.txt


module load miniconda3
source activate DM



export DATADIR="/work/ws-tmp/ms152708-DeepMend/test_shapeNet/ShapeNet"

export PYTHONPATH="/work/ws-tmp/ms152708-DeepMend/DeepMend_changes3_new2/fracturing"




EXPERIMENT="/work/ws-tmp/ms152708-DeepMend/DeepMend_changes3_new2/deepmend/experiments/cars"     # Path to an experiment to evaluate
CHK="latest"
UNIFORM_RATIO="0.2"
NUMITS="3000"
LREG="0.000"
LR="0.005"
LAMBDANER="0.00001"
LAMBDAPROX="0.000"
NME="deepmend"

echo "Reconstructing $NME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

python python/reconstruct_1.py \
    -e "$EXPERIMENT" \
    -c "$CHK" \
    --name "$NME" \
    --threads 5 \
    --num_iters "$NUMITS" \
    --lambda_reg "$LREG" \
    --learning_rate "$LR" \
    --render_threads 5 \
    --uniform_ratio "$UNIFORM_RATIO" \
    --lambda_ner "$LAMBDANER" \
    --lambda_prox "$LAMBDAPROX" \
    --stop 100

