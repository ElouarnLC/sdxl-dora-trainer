#!/bin/bash
"""
Full Enhanced Training Launcher
Optimized for maximum performance on your multihead dataset
"""

echo "🚀 Starting Full Enhanced Training"
echo "📊 Dataset: 125 training samples, 31 validation samples"
echo "🎯 Target: Break F1 0.65-0.66 plateau → achieve F1 0.75-0.85+"
echo ""

# Full enhanced training optimized for your dataset size
python scripts/train_small_dataset.py \
    --data data/annotations/multihead_ratings.csv \
    --prompts data/prompts.csv \
    --output outputs/full_enhanced_final \
    --epochs 50 \
    --lr 2e-4 \
    --batch-size 4

echo ""
echo "✅ Training completed!"
echo "📁 Models saved to: outputs/full_enhanced_final/"
echo "🏆 Best model: outputs/full_enhanced_final/best/model.pt"
echo "📊 Check training logs for final F1 score"
