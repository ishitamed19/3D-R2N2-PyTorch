#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

NET_NAME=ResidualGRUNet
EXP_DETAIL=baseline
OUT_PATH='./output/'$NET_NAME/$EXP_DETAIL
TB_PATH='./tb_output/'$NET_NAME/$EXP_DETAIL
LOG="$OUT_PATH/log.`date +'%Y-%m-%d_%H-%M-%S'`"

# Make the dir if it not there
mkdir -p $OUT_PATH
mkdir -p $TB_PATH
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python3 main.py \
      --batch-size 1 \
      --iter 100000 \
      --lr 1e-4\
      --dict False \
      --out $OUT_PATH \
      --model $NET_NAME \
      --tb $TB_PATH/'train' \
      ${*:1}

# python3 main.py \
#       --test \
#       --batch-size 1 \
#       --out $OUT_PATH \
#       --weights $OUT_PATH/checkpoint.pth \
#       --model $NET_NAME \
#       ${*:1}
