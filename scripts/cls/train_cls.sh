#!/bin/bash
DATA_PATH='data/A1_clip'
MODEL_PATH='data/checkpoint.pth'
OUTPUT_DIR='./checkpoints/cobot/'
CUDA_VISIBLE_DEVICES=1 python run_class_finetuning.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 20 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 2 \
    --num_frames 8 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.999 \
    --warmup_epochs 5 \
    --epochs 25 \
    --test_num_segment 10 \
    --test_num_crop 1 \
    --lr 1e-3 \
    --layer_decay 0.75 \
    --dist_eval