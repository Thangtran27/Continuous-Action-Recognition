
# Set the path to save checkpoints
OUTPUT_DIR='pickles/cobot'
DATA_PATH='data/video_ids.txt'

MODEL_PATH='./checkpoints/cobot/checkpoint-24.pth'
CUDA_VISIBLE_DEVICES=1 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 20 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --csv_file ${DATA_PATH} \
    --batch_size 2 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 25 \
    --lr 2e-3 \
    --clip_stride 30 \
    --crop \