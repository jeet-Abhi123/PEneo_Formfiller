#!/bin/bash
# Train ONE model sequentially across all folds using 5-fold cross-validation
# Model accumulates knowledge from each fold
# Total: 601 files -> 481 train, 120 val per fold (5-fold CV)
# Each fold: 3000 steps
# Total steps: 15000 steps (5 folds × 3000 steps)

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS='true'

# Get number of GPUs
PROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
MASTER_PORT=11451

# Configuration
TASK_NAME=layoutlmv3-base-sibr
PRETRAINED_PATH=private_pretrained/layoutlmv3-base
BASE_DATA_DIR=/home/abhij/nhance_form_filler/PEneo_group/private_data_13_primary/sibr
CV_FOLDS_DIR=$BASE_DATA_DIR/cv_folds
OUTPUT_DIR=private_output_13_primary/weights/$TASK_NAME
RUNS_DIR=private_output_13_primary/runs/$TASK_NAME
LOG_DIR=private_output_13_primary/logs/$TASK_NAME.log
FINAL_EVAL_DIR=private_output_13_primary/final_eval

# Training parameters per fold (601 files -> 481 train, 120 val per fold)
STEPS_PER_FOLD=3000  # Steps to train on each fold
EVAL_STEPS=500
SAVE_STEPS=500
N_FOLDS=5
TRAIN_SIZE=481
VAL_SIZE=120

# Create output directories
mkdir -p private_output_13_primary/weights
mkdir -p private_output_13_primary/logs
mkdir -p private_output_13_primary/runs
mkdir -p $FINAL_EVAL_DIR
mkdir -p $CV_FOLDS_DIR

echo "=========================================="
echo "Sequential Training Across All Folds (5-Fold CV)"
echo "=========================================="
echo "Task: $TASK_NAME"
echo "Pre-trained model: $PRETRAINED_PATH"
echo "Base data directory: $BASE_DATA_DIR"
echo "CV folds directory: $CV_FOLDS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of folds: $N_FOLDS"
echo "Steps per fold: $STEPS_PER_FOLD"
echo "Total steps: $((STEPS_PER_FOLD * N_FOLDS))"
echo "Train size per fold: $TRAIN_SIZE"
echo "Val size per fold: $VAL_SIZE"
echo "Batch size: 2"
echo "Learning rate: 1e-5"
echo "Using $PROC_PER_NODE GPU(s)"
echo "=========================================="

# Ensure base data has train.txt (and optionally test.txt) for create_cv_folds
if [ ! -f "$BASE_DATA_DIR/train.txt" ] && [ ! -f "$BASE_DATA_DIR/test.txt" ]; then
    echo "Error: Base data directory must contain train.txt or test.txt listing file names."
    echo "  Missing: $BASE_DATA_DIR/train.txt and $BASE_DATA_DIR/test.txt"
    exit 1
fi

# Step 1: Create CV folds if they don't exist or are incomplete (missing train.txt)
if [ ! -f "$CV_FOLDS_DIR/fold_1/train.txt" ]; then
    echo ""
    echo "Creating cross-validation folds..."
    [ -d "$CV_FOLDS_DIR" ] && rm -rf "${CV_FOLDS_DIR:?}"/*
    python create_cv_folds.py \
        --data_dir "$BASE_DATA_DIR" \
        --output_dir "$CV_FOLDS_DIR" \
        --n_folds $N_FOLDS \
        --train_size $TRAIN_SIZE \
        --val_size $VAL_SIZE
    echo ""
fi

# Step 2: Train sequentially across all folds
CURRENT_MODEL_PATH=$PRETRAINED_PATH
TOTAL_STEPS=0
ALL_EVAL_RESULTS=()

for fold in $(seq 1 $N_FOLDS); do
    echo ""
    echo "=========================================="
    echo "Training on Fold $fold / $N_FOLDS"
    echo "=========================================="
    
    FOLD_DATA_DIR=$CV_FOLDS_DIR/fold_$fold
    FOLD_LOG_FILE=$LOG_DIR.fold_$fold
    
    echo "Fold $fold data directory: $FOLD_DATA_DIR"
    echo "Fold $fold log: $FOLD_LOG_FILE"
    echo "Loading model from: $CURRENT_MODEL_PATH"
    echo "Training for $STEPS_PER_FOLD steps on this fold"
    
    # Calculate cumulative steps
    TOTAL_STEPS=$((TOTAL_STEPS + STEPS_PER_FOLD))
    
    # Run training for this fold
    # Always use original pretrained model for config (to preserve backbone_name)
    # But use checkpoint for model weights
    if [ $PROC_PER_NODE -gt 1 ]; then
        # Multi-GPU training
        torchrun --nproc_per_node $PROC_PER_NODE --master_port $MASTER_PORT start/run_sibr.py \
            --model_name_or_path $CURRENT_MODEL_PATH \
            --config_name $PRETRAINED_PATH \
            --data_dir $FOLD_DATA_DIR \
            --output_dir $OUTPUT_DIR \
            --do_train \
            --do_eval \
            --fp16 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 16 \
            --dataloader_num_workers 4 \
            --warmup_ratio 0.1 \
            --learning_rate 5e-5 \
            --max_steps $TOTAL_STEPS \
            --eval_strategy steps \
            --eval_steps $EVAL_STEPS \
            --save_strategy steps \
            --save_steps $SAVE_STEPS \
            --load_best_model_at_end False \
            --save_total_limit 2 \
            --logging_strategy steps \
            --logging_steps 100 \
            --logging_dir $RUNS_DIR \
            --detail_eval True \
            --save_eval_detail True \
            2>&1 | tee -a $FOLD_LOG_FILE
    else
        # Single GPU training
        python start/run_sibr.py \
            --model_name_or_path $CURRENT_MODEL_PATH \
            --config_name $PRETRAINED_PATH \
            --data_dir $FOLD_DATA_DIR \
            --output_dir $OUTPUT_DIR \
            --do_train \
            --do_eval \
            --fp16 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 8 \
            --dataloader_num_workers 4 \
            --warmup_ratio 0.1 \
            --learning_rate 5e-5 \
            --max_steps $TOTAL_STEPS \
            --eval_strategy steps \
            --eval_steps $EVAL_STEPS \
            --save_strategy steps \
            --save_steps $SAVE_STEPS \
            --load_best_model_at_end False \
            --save_total_limit 1 \
            --logging_strategy steps \
            --logging_steps 100 \
            --logging_dir $RUNS_DIR \
            --detail_eval True \
            --save_eval_detail True \
            2>&1 | tee -a $FOLD_LOG_FILE
    fi
    
    # Update model path for next fold (use latest checkpoint from current fold)
    CURRENT_MODEL_PATH=$OUTPUT_DIR
    
    # Extract eval results from log
    if [ -f "$OUTPUT_DIR/eval_results.json" ]; then
        echo ""
        echo "Fold $fold evaluation results:"
        python3 -c "
import json
with open('$OUTPUT_DIR/eval_results.json') as f:
    results = json.load(f)
    print(f\"  F1 Score: {results.get('eval_f1', 'N/A')}\")
    print(f\"  Precision: {results.get('eval_precision', 'N/A')}\")
    print(f\"  Recall: {results.get('eval_recall', 'N/A')}\")
"
        # Copy eval results
        cp $OUTPUT_DIR/eval_results.json $FINAL_EVAL_DIR/fold_${fold}_eval_results.json
        if [ -f "$OUTPUT_DIR/detail.json" ]; then
            cp $OUTPUT_DIR/detail.json $FINAL_EVAL_DIR/fold_${fold}_detail.json
        fi
    fi
    
    echo ""
    echo "Fold $fold training completed!"
    echo "Model updated and ready for next fold"
    echo ""
done

# Step 3: Final evaluation on all validation sets combined
echo "=========================================="
echo "Final Evaluation on All Validation Sets"
echo "=========================================="

# Combine all validation files
COMBINED_VAL_DIR=$FINAL_EVAL_DIR/combined_val
mkdir -p $COMBINED_VAL_DIR

# Copy all validation files to combined directory
cat $CV_FOLDS_DIR/fold_*/test.txt | sort -u > $COMBINED_VAL_DIR/test.txt

# Copy corresponding JSON files (assuming they're in the base data dir)
echo "Creating combined validation set..."
python3 << PYEOF
import os
from pathlib import Path

cv_folds_dir = Path('$CV_FOLDS_DIR')
base_data_dir = Path('$BASE_DATA_DIR').resolve()
combined_val_dir = Path('$COMBINED_VAL_DIR')

# Read all validation files
all_val_files = set()
for fold_dir in sorted(cv_folds_dir.glob('fold_*')):
    val_file = fold_dir / 'test.txt'
    if val_file.exists():
        with open(val_file) as f:
            all_val_files.update([line.strip() for line in f if line.strip()])

# Write combined validation file
with open(combined_val_dir / 'test.txt', 'w') as f:
    for file in sorted(all_val_files):
        f.write(file + '\n')

print(f"Combined validation set: {len(all_val_files)} unique files")

# Create symlinks to converted_label and images directories
converted_label_link = combined_val_dir / 'converted_label'
images_link = combined_val_dir / 'images'

# Remove existing symlinks if they exist
if converted_label_link.exists() or converted_label_link.is_symlink():
    converted_label_link.unlink()
if images_link.exists() or images_link.is_symlink():
    images_link.unlink()

# Create symlinks with absolute paths
target_converted = base_data_dir / 'converted_label'
target_images = base_data_dir / 'images'

if target_converted.exists():
    converted_label_link.symlink_to(target_converted)
    print(f"Created symlink: {converted_label_link} -> {target_converted}")

if target_images.exists():
    images_link.symlink_to(target_images)
    print(f"Created symlink: {images_link} -> {target_images}")
PYEOF

# Final evaluation
echo ""
echo "Running final evaluation on combined validation set..."
if [ $PROC_PER_NODE -gt 1 ]; then
    torchrun --nproc_per_node $PROC_PER_NODE --master_port $MASTER_PORT start/run_sibr.py \
        --model_name_or_path $OUTPUT_DIR \
        --config_name $PRETRAINED_PATH \
        --data_dir $COMBINED_VAL_DIR \
        --output_dir $FINAL_EVAL_DIR \
        --do_eval \
        --fp16 \
        --per_device_eval_batch_size 16 \
        --dataloader_num_workers 4 \
        --detail_eval True \
        --save_eval_detail True \
        2>&1 | tee -a $LOG_DIR.final_eval
else
    python start/run_sibr.py \
        --model_name_or_path $OUTPUT_DIR \
        --config_name $PRETRAINED_PATH \
        --data_dir $COMBINED_VAL_DIR \
        --output_dir $FINAL_EVAL_DIR \
        --do_eval \
        --fp16 \
        --per_device_eval_batch_size 16 \
        --dataloader_num_workers 4 \
        --detail_eval True \
        --save_eval_detail True \
        2>&1 | tee -a $LOG_DIR.final_eval
fi

# Step 4: Summary
echo ""
echo "=========================================="
echo "Training Summary"
echo "=========================================="
echo "Final model weights: $OUTPUT_DIR"
echo "Final evaluation results: $FINAL_EVAL_DIR/eval_results.json"
echo "Per-fold evaluation results: $FINAL_EVAL_DIR/fold_*_eval_results.json"
echo ""
echo "Final evaluation metrics:"
if [ -f "$FINAL_EVAL_DIR/eval_results.json" ]; then
    python3 -c "
import json
with open('$FINAL_EVAL_DIR/eval_results.json') as f:
    results = json.load(f)
    print(f\"  F1 Score: {results.get('eval_f1', 'N/A')}\")
    print(f\"  Precision: {results.get('eval_precision', 'N/A')}\")
    print(f\"  Recall: {results.get('eval_recall', 'N/A')}\")
"
fi
echo ""
echo "All training logs: $LOG_DIR.fold_*"
echo "Tensorboard: tensorboard --logdir $RUNS_DIR"