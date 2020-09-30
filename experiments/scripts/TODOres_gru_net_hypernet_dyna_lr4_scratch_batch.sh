#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

NET_NAME=ResidualGRUNetHypernet
EXP_DETAIL=dyna_lr4_scratch_bs1_vq100
OUT_PATH='./output/'$NET_NAME/$EXP_DETAIL
TB_PATH='./tb_output/'$NET_NAME/$EXP_DETAIL
LOG="$OUT_PATH/log.`date +'%Y-%m-%d_%H-%M-%S'`"

#also increasing weightage for vqloss
#__C.TRAIN.LEARNING_RATES = {'75000': 5e-5, '125000': 1e-5}
# Make the dir if it not there
mkdir -p $OUT_PATH
mkdir -p $TB_PATH
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python3 main.py \
      --batch-size 1 \
      --iter 200000 \
      --lr 1e-4 \
      --dict True \
      --out $OUT_PATH \
      --net 'res_gru_net_hypernet' \
      --model $NET_NAME \
      --tb $TB_PATH/'train' \
      ${*:1}

# python3 main.py \
#       --test \
#       --batch-size 1 \
#       --out $OUT_PATH \
# #       --weights $OUT_PATH/checkpoint.pth \
#       --model $NET_NAME \
#       --tb $TB_PATH/'test' \
#       ${*:1}
