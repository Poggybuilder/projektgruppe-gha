#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python train_base.py \
    --dataset-name backgrounds \
    --model-backbone resnet50 \
    --model-name mattingbase-resnet50-videomatte240k \
    --model-pretrain-initialization "pretraining/vci.pth" \
    --epoch-end 8
