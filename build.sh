#!/bin/bash
set -e

echo "Installing build tools..."
pip install --upgrade pip setuptools wheel

echo "Installing requirements..."
pip install -r requirements.txt

echo "Build complete!"