#!/bin/bash
set -o errexit

# Install dependencies with pip
pip install --upgrade pip
pip install -r requirements.txt

echo "Build completed successfully!"
