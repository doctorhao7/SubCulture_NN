#!/bin/bash
set -o errexit

echo "==> Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Training PyTorch model..."
python3 train_pytorch.py

echo "==> Build completed successfully!"
