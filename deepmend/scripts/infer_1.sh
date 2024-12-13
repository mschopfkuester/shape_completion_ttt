# !/bin/bash
# Perform inference using deepmend
#

EXPERIMENT="$1"     # Path to an experiment to evaluate
CHK="2000"
UNIFORM_RATIO="0.2"
NUMITS="3000"
LREG="0.000"
LR="0.01"
LAMBDANER="0.00001"
LAMBDAPROX="0.00"
NME="deepmend"

echo "Loading from ""$1"
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
    --lambda_prox "$LAMBDAPROX"

