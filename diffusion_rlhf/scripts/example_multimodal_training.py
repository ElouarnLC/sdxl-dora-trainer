#!/usr/bin/env python3
"""
Example script demonstrating multimodal reward model training usage.

This script shows how to train a multimodal reward model that processes
both images and prompts for enhanced prompt-image alignment assessment.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Demonstrate multimodal reward model training."""
    
    print("=== Multimodal Reward Model Training Examples ===\n")
    
    # Base paths
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data"
    
    # Check required files
    prompts_file = data_path / "prompts.csv"
    ratings_file = data_path / "annotations" / "multihead_ratings.csv"
    
    if not prompts_file.exists():
        print(f"ERROR: Prompts file not found: {prompts_file}")
        print("Please ensure prompts.csv exists in data/ directory")
        return 1
    
    if not ratings_file.exists():
        print(f"WARNING: Ratings file not found: {ratings_file}")
        print("Multihead training example will be skipped")
    
    print("Found required files:")
    print(f"  Prompts: {prompts_file}")
    if ratings_file.exists():
        print(f"  Ratings: {ratings_file}")
    print()
    
    # Example 1: Basic multimodal training (requires pairs.jsonl)
    print("1. Basic Multimodal Training (requires pairs.jsonl):")
    print("   This mode trains on preference pairs with prompt context")
    print()
    
    pairs_example = """
    python scripts/train_multimodal_reward.py \\
        --mode pairs \\
        --pairs data/pairs.jsonl \\
        --prompts data/prompts_example.csv \\
        --output outputs/multimodal_reward_basic \\
        --epochs 5 \\
        --batch-size 4 \\
        --fusion-method concat \\
        --mixed-precision fp16
    """
    print("   Command:")
    print(pairs_example)
    print()
    
    # Example 2: Multi-head training with different fusion methods
    if ratings_file.exists():
        print("2. Multi-head Multimodal Training:")
        print("   This mode trains on multi-aspect ratings with prompt context")
        print()
        
        multihead_examples = {
            "Concatenation Fusion": "concat",
            "Addition Fusion": "add", 
            "Cross-Attention Fusion": "cross_attention"
        }
        
        for name, fusion_method in multihead_examples.items():
            print(f"   {name}:")
            command = f"""
    python scripts/train_multimodal_reward.py \\
        --mode multihead \\
        --ratings {ratings_file} \\
        --prompts {prompts_file} \\
        --output outputs/multimodal_reward_{fusion_method} \\
        --epochs 10 \\
        --batch-size 8 \\
        --fusion-method {fusion_method} \\
        --min-rating-diff 1.0 \\
        --learning-rate 1e-4 \\
        --mixed-precision bf16
            """
            print("   Command:")
            print(command)
            print()
    
    # Key improvements explanation
    print("=== Key Improvements in Multimodal Training ===")
    print()
    print("1. Enhanced Prompt-Image Alignment:")
    print("   - Model considers both visual content and text prompt")
    print("   - Better assessment of how well images match their prompts")
    print("   - Critical for prompt-conditioned generation evaluation")
    print()
    
    print("2. Multiple Fusion Strategies:")
    print("   - concat: Simple concatenation of image and text features")
    print("   - add: Element-wise addition (requires same dimensionality)")
    print("   - cross_attention: Advanced attention-based fusion")
    print()
    
    print("3. Specialized Head Training:")
    print("   - spatial: Composition and layout with prompt context")
    print("   - icono: Iconographic elements matching prompt descriptions")
    print("   - style: Artistic style alignment with prompt style cues")
    print("   - fidelity: Image quality in context of prompt complexity")
    print("   - material: Material properties as described in prompts")
    print()
    
    print("4. Evaluation Metrics:")
    print("   - Per-head F1 scores for fine-grained evaluation")
    print("   - Combined model performance across all aspects")
    print("   - Head importance weights showing learned priorities")
    print()
    
    # Usage recommendations
    print("=== Usage Recommendations ===")
    print()
    print("• Start with 'concat' fusion for simplicity and reliability")
    print("• Use 'cross_attention' for more sophisticated prompt-image interaction")
    print("• Adjust min_rating_diff based on your annotation granularity")
    print("• Monitor per-head metrics to understand model specialization")
    print("• Use mixed precision (fp16/bf16) for faster training")
    print()
    
    # Quick validation
    print("=== Quick Validation ===")
    print("To test the multimodal components without full training:")
    print()
    validation_command = """
    python -c "
from dspo.multimodal_reward import MultimodalMultiHeadReward
from dspo.multimodal_datasets import MultimodalPairPreferenceDataset
import torch

# Test model initialization
model = MultimodalMultiHeadReward(fusion_method='concat')
print('✓ Multimodal model created successfully')

# Test forward pass
dummy_images = torch.randn(2, 3, 224, 224)
dummy_prompts = ['A beautiful sunset', 'A cute cat']
outputs = model(dummy_images, dummy_prompts)
print(f'✓ Forward pass successful, output shape: {outputs.shape}')

print('✓ All multimodal components working correctly!')
"
    """
    print(validation_command)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
