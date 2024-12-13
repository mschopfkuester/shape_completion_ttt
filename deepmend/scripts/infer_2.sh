# !/bin/bash
# Perform inference using deepmend
#
#export DATADIR="/home/michael/Projects/test_shapeNet_inference2/ShapeNet"


#cd ../own_code
#python latent_code.py

#cd ../deepmend
#export PYTHONPATH="../fracturing";python python/train.py -e experiments/mugs -c b





export DATADIR="/home/michael/Projects/test_shapeNet/ShapeNet"

EXPERIMENT="$1"     # Path to an experiment to evaluate
CHK="latest"
UNIFORM_RATIO="0.2"
NUMITS="0"
LREG="0.0001"
LR="0.01"
LAMBDANER="0.00001"
LAMBDAPROX="0.005"
NME="deepmend"

echo "Loading from ""$1"
echo "Reconstructing $NME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

python python/reconstruct_2.py \
    -e "$EXPERIMENT" \
    -c "$CHK" \
    --name "$NME" \
    --threads 2 \
    --num_iters "$NUMITS" \
    --lambda_reg "$LREG" \
    --learning_rate "$LR" \
    --render_threads 2 \
    --uniform_ratio "$UNIFORM_RATIO" \
    --lambda_ner "$LAMBDANER" \
    --lambda_prox "$LAMBDAPROX"

