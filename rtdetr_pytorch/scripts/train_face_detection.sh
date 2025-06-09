#!/bin/bash
# scripts/train_face_detection.sh
# Training script for RT-DETR face detection with landmarks

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Configuration
CONFIG="configs/rtdetr/rtdetr_r50vd_face_landmark_improved.yml"
OUTPUT_DIR="./output/rtdetr_r50vd_face_landmark_v2"
LOG_FILE="${OUTPUT_DIR}/training.log"

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "Starting RT-DETR Face Detection Training..."
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"

# Check dataset first
echo ""
echo "Checking training setup..."
python tools/check_training_setup.py --config ${CONFIG}

if [ $? -ne 0 ]; then
    echo "Setup check failed! Please fix issues before training."
    exit 1
fi

echo ""
echo "Setup check passed. Starting training..."
echo "Training will be logged to: ${LOG_FILE}"
echo ""

# Start training with pretrained COCO weights
python tools/train_polar_landmarks.py \
    --config ${CONFIG} \
    --tuning https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth \
    --amp \
    --val-freq 5 \
    --seed 42 \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "Training completed!"
echo "Model saved to: ${OUTPUT_DIR}"