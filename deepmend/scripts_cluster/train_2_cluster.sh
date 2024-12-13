#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=17
#SBATCH --time=03:00:00

#SBATCH --array=1-4%4





#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=michael.schopf@uni-siegen.de
#SBATCH --output=/work/ws-tmp/ms152708-DeepMend/DeepMend_changes3_new2/output_files_cluster/jars/train_2/output%a.txt
#SBATCH --error=/work/ws-tmp/ms152708-DeepMend/DeepMend_changes3_new2/output_files_cluster/jars/train_2/error%a.txt
module load miniconda3
source activate DM





export PYTHONPATH="/work/ws-tmp/ms152708-DeepMend/DeepMend_changes3_new2/fracturing"



DATADIR1="/work/ws-tmp/ms152708-DeepMend/test_shapeNet/ShapeNet/ShapeNetCore.v2"
DATADIR2="/work/ws-tmp/ms152708-DeepMend/test_shapeNet_inference2/ShapeNet/ShapeNetCore.v2"
TrainTestFile="/work/ws-tmp/ms152708-DeepMend/test_shapeNet/ShapeNet/ShapeNetCore.v2/jars_split.json"
modelpath="/work/ws-tmp/ms152708-DeepMend/Model_1000/jars.pth"
object_class="jars"

python python/train_2_all_cluster.py \
    --datadir_1 "$DATADIR1" \
    --datadir_2 "$DATADIR2" \
    --train_test_file "$TrainTestFile" \
    --k_obj $SLURM_ARRAY_TASK_ID \
    --obj_class "$object_class" \
    --path_model "$modelpath"