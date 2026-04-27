#!/bin/bash
echo "Starting data preprocessing..."
python3 scripts/preprocess_data.py --sample-size 4000

echo "Starting model fine-tuning..."
python3 scripts/train.py --config configs/train.yaml

echo "Training complete!"