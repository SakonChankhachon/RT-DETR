#!/bin/bash
# RT-DETR Face Detection Training Script
# This script runs face landmark detection training with the improved configuration

# Exit on error
set -e

# Configuration
CONFIG_FILE="configs/rtdetr/rtdetr_r50vd_face_landmark_improved.yml"
OUTPUT_DIR="./output/rtdetr_r50vd_face_landmark_v2"
LOG_FILE="$OUTPUT_DIR/training.log"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print header
echo "Starting RT-DETR Face Detection Training..."
echo ""
echo "Config: $CONFIG_FILE"
echo ""
echo "Output: $OUTPUT_DIR"
echo ""
echo "Log: $LOG_FILE"
echo ""

# Check if dataset exists
if [ ! -d "./dataset/faces" ]; then
    echo "❌ Error: Dataset directory not found."
    echo "Please run setup_face_detection.sh first to prepare the dataset."
    exit 1
fi

echo "Checking training setup..."
echo ""

# Try to install ujson for faster annotation loading
pip install ujson 2>/dev/null || echo "NOTE! Installing ujson may make loading annotations faster."

# Check if face landmark components are available
python -c "from src.zoo.rtdetr.rtdetr_face_decoder import RTDETRTransformerPolarLandmark; print('✅ Face landmark components loaded successfully')" || {
    echo "❌ Error: Face landmark components not found."
    exit 1
}

# Run detailed setup check
echo "Checking training setup for: $CONFIG_FILE"
echo ""
python tools/debug_dataset.py -c "$CONFIG_FILE" || {
    echo "❌ Error: Dataset check failed."
    echo "Please verify your dataset configuration."
    exit 1
}

echo "✅ All checks passed! Ready to train."
echo ""
echo "To start training, run:"
echo "  python tools/train_polar_landmarks.py -c $CONFIG_FILE"
echo ""

echo "Setup check passed. Starting training..."
echo ""
echo "Training will be logged to: $LOG_FILE"

# Run the training
python tools/train_polar_landmarks.py -c "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed!"
    echo "Model saved to: $OUTPUT_DIR"
    exit 0
else
    echo "❌ Training failed! Check the log file for details: $LOG_FILE"
    exit 1
fi
