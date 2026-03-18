#!/bin/bash
# Run inference with trained PEneo model on all images in a directory

export PYTHONPATH=./

# Configuration
TASK_NAME=layoutlmv3-base-sibr
MODEL_PATH=private_output_13_primary/weights/$TASK_NAME

# Directory containing images and OCR JSON files
# OCR files should be named as {image_name}_ocr.json
INPUT_DIR="/home/abhij/nhance_form_filler/testing_images_6"
VISUALIZE_PATH=/home/abhij/nhance_form_filler/outputs/Primary_Group_test_ocr

mkdir -p "$VISUALIZE_PATH"

echo "Starting inference..."
echo "Model: $MODEL_PATH"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $VISUALIZE_PATH"
echo "Processing images one at a time to avoid memory issues..."
echo "----------------------------------------"

# Collect all image files first
IMAGE_FILES=()
OCR_FILES=()

# Enable nullglob to handle cases where no files match
shopt -s nullglob

# Collect image files with different extensions
for ext in png jpg jpeg PNG JPG JPEG; do
    for img_file in "$INPUT_DIR"/*.$ext; do
        if [ -f "$img_file" ]; then
            img_basename=$(basename "$img_file")
            img_name="${img_basename%.*}"
            ocr_file="$INPUT_DIR/${img_name}_ocr.json"
            
            if [ -f "$ocr_file" ]; then
                IMAGE_FILES+=("$img_file")
                OCR_FILES+=("$ocr_file")
            else
                echo "Warning: OCR file not found for $img_basename, skipping..."
            fi
        fi
    done
done

# Disable nullglob
shopt -u nullglob

TOTAL_IMAGES=${#IMAGE_FILES[@]}

if [ $TOTAL_IMAGES -eq 0 ]; then
    echo "Error: No image files with matching OCR files found in $INPUT_DIR"
    exit 1
fi

echo "Found $TOTAL_IMAGES image file(s) to process"
echo ""

# Create temporary directories for all images
TEMP_DIR=$(mktemp -d)
TEMP_IMAGE_DIR="$TEMP_DIR/images"
TEMP_OCR_DIR="$TEMP_DIR/ocr"

mkdir -p "$TEMP_IMAGE_DIR"
mkdir -p "$TEMP_OCR_DIR"

# Copy all images and OCR files to temp directories
echo "Preparing images..."
for i in "${!IMAGE_FILES[@]}"; do
    img_file="${IMAGE_FILES[$i]}"
    ocr_file="${OCR_FILES[$i]}"
    
    cp "$img_file" "$TEMP_IMAGE_DIR/"
    cp "$ocr_file" "$TEMP_OCR_DIR/"
done

# Cleanup function
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

echo "Running inference (model will be loaded once, images processed sequentially)..."
echo ""

# Run inference once - model loads once, images processed sequentially via generator
START_TIME=$(date +%s)
python deploy/inference.py \
  --model_name_or_path "$MODEL_PATH" \
  --dir_image "$TEMP_IMAGE_DIR" \
  --dir_ocr "$TEMP_OCR_DIR" \
  --visualize_path "$VISUALIZE_PATH"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Inference completed successfully!"
    echo "Processed: $TOTAL_IMAGES images"
    echo "Total time: ${ELAPSED}s"
    echo "Average time per image: $(echo "scale=2; $ELAPSED / $TOTAL_IMAGES" | bc)s"
else
    echo "Inference failed with exit code: $EXIT_CODE"
fi
echo "Results saved to: $VISUALIZE_PATH"
echo "=========================================="

