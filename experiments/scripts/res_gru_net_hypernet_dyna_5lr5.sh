#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

NET_NAME=ResidualGRUNetHypernet
EXP_DETAIL=dyna_5lr5_resumed
OUT_PATH='./output/'$NET_NAME/$EXP_DETAIL
TB_PATH='./tb_output/'$NET_NAME/$EXP_DETAIL
LOG="$OUT_PATH/log.`date +'%Y-%m-%d_%H-%M-%S'`"

#__C.TRAIN.LEARNING_RATES = {'50000': 1e-5, '150000': 5e-6}
# Make the dir if it not there
mkdir -p $OUT_PATH
mkdir -p $TB_PATH
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python3 main.py \
      --batch-size 1 \
      --iter 90100 \
      --lr 5e-5 \
      --dict True \
      --out $OUT_PATH \
      --net 'res_gru_net_hypernet' \
      --weights output/ResidualGRUNetHypernet/dyna_5lr5/checkpoint.40000.pth \
      --init-iter 40000 \
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
