#!/bin/bash

set -e

echo "JAIDE v40 Modal Setup"
echo "====================="

if ! command -v pip &> /dev/null; then
    echo "ERROR: pip not found. Install Python first."
    exit 1
fi

if ! command -v modal &> /dev/null; then
    echo "Installing Modal CLI..."
    pip install -q modal
fi

echo "Checking Modal authentication..."
if ! modal token check &> /dev/null; then
    echo ""
    echo "Modal token not found. Please authenticate:"
    echo "  modal token new"
    echo ""
    echo "This will open a browser window to log in to modal.com."
    modal token new
fi

echo ""
echo "Creating Modal volumes for training data and checkpoints..."
modal volume create jaide-training-data 2>/dev/null || echo "  Volume 'jaide-training-data' already exists."
modal volume create jaide-dataset 2>/dev/null || echo "  Volume 'jaide-dataset' already exists."

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run training (8x B200 GPUs):"
echo "       cd jaide/src/scripts && modal run modal_train.py"
echo ""
echo "  2. Run with custom parameters:"
echo "       modal run modal_train.py --epochs 100 --dim 1024 --layers 12"
echo ""
echo "  3. Run inference after training:"
echo "       modal run modal_inference.py --prompt 'Your prompt here'"
echo ""
echo "  4. List saved models:"
echo "       modal run modal_train.py::list_models"
