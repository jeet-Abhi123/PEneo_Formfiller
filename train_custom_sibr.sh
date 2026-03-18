#!/bin/bash
# Fine-tune PEneo model on custom SIBR dataset

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS='true'

# Get number of GPUs
PROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
MASTER_PORT=11451

# Configuration
TASK_NAME=layoutlmv3-base-sibr
PRETRAINED_PATH=private_pretrained/layoutlmv3-base
DATA_DIR=private_data_primary_grp_10/sibr   #/sibr
OUTPUT_DIR=private_output_primary_grp_10/weights/$TASK_NAME
RUNS_DIR=private_output_primary_grp_10/runs/$TASK_NAME
LOG_DIR=private_output_primary_grp_10/logs/$TASK_NAME.log

# Create output directories
mkdir -p private_output_primary_grp_10/weights
mkdir -p private_output_primary_grp_10/logs
mkdir -p private_output_primary_grp_10/runs

# Training parameters
# Adjusted for smaller dataset (19 files total)
# With 90% train split = ~17 files, 10% test = ~2 files
# Reduced max_steps since dataset is smaller
MAX_STEPS=12000
EVAL_STEPS=500
SAVE_STEPS=500  # Must match eval_steps when using load_best_model_at_end

echo "Starting fine-tuning..."
echo "Task: $TASK_NAME"
echo "Pre-trained model: $PRETRAINED_PATH"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Max steps: $MAX_STEPS"
echo "Using $PROC_PER_NODE GPU(s)"
echo "----------------------------------------"

# Run training
if [ $PROC_PER_NODE -gt 1 ]; then
    # Multi-GPU training
    torchrun --nproc_per_node $PROC_PER_NODE --master_port $MASTER_PORT start/run_sibr.py \
        --model_name_or_path $PRETRAINED_PATH \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --do_train \
        --do_eval \
        --fp16 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 16 \
        --dataloader_num_workers 4 \
        --warmup_ratio 0.2 \
        --learning_rate 3e-5 \
        --max_steps $MAX_STEPS \
        --eval_strategy steps \
        --eval_steps $EVAL_STEPS \
        --save_strategy steps \
        --save_steps $SAVE_STEPS \
        --load_best_model_at_end True \
        --metric_for_best_model f1 \
        --save_total_limit 1 \
        --logging_strategy steps \
        --logging_steps 100 \
        --logging_dir $RUNS_DIR \
        --detail_eval True \
        --save_eval_detail True \
        2>&1 | tee -a $LOG_DIR
else
    # Single GPU training
    python start/run_sibr.py \
        --model_name_or_path $PRETRAINED_PATH \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --do_train \
        --do_eval \
        --fp16 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 16 \
        --dataloader_num_workers 4 \
        --warmup_ratio 0.2 \
        --learning_rate 2e-5 \
        --max_steps $MAX_STEPS \
        --eval_strategy steps \
        --eval_steps $EVAL_STEPS \
        --save_strategy steps \
        --save_steps $SAVE_STEPS \
        --load_best_model_at_end True \
        --metric_for_best_model f1 \
        --save_total_limit 1 \
        --logging_strategy steps \
        --logging_steps 100 \
        --logging_dir $RUNS_DIR \
        --detail_eval True \
        --save_eval_detail True \
        2>&1 | tee -a $LOG_DIR
fi

echo "Training completed!"
echo "Check logs at: $LOG_DIR"
echo "Check tensorboard: tensorboard --logdir $RUNS_DIR"
echo "Model saved at: $OUTPUT_DIR"