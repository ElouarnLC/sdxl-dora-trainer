#!/usr/bin/env python3
"""
Data validation script to check for issues with input images.
"""

import torch
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from torchvision import transforms

def validate_image_dataset(dataset_path: str, max_images: int = 10):
    """Validate images in the dataset for potential issues."""
    
    print(f"Validating dataset at: {dataset_path}")
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return False
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(dataset_path.rglob(f'*{ext}')))
        image_files.extend(list(dataset_path.rglob(f'*{ext.upper()}')))
    
    if not image_files:
        print("ERROR: No image files found in dataset")
        return False
    
    print(f"Found {len(image_files)} image files")
    
    # Setup transforms (same as in training)
    transforms_list = transforms.Compose([
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    issues_found = 0
    
    for i, image_path in enumerate(image_files[:max_images]):
        print(f"\nValidating image {i+1}/{min(max_images, len(image_files))}: {image_path.name}")
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            print(f"  Original size: {image.size}")
            
            # Check for valid size
            if image.size[0] == 0 or image.size[1] == 0:
                print(f"  ERROR: Invalid image size: {image.size}")
                issues_found += 1
                continue
            
            # Apply transforms
            tensor = transforms_list(image)
            print(f"  Tensor shape: {tensor.shape}")
            print(f"  Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
            
            # Check for NaN or inf
            if torch.isnan(tensor).any():
                print(f"  ERROR: NaN values found in tensor")
                issues_found += 1
            
            if torch.isinf(tensor).any():
                print(f"  ERROR: Inf values found in tensor")
                issues_found += 1
            
            # Check value range
            if tensor.min() < -10 or tensor.max() > 10:
                print(f"  WARNING: Extreme values in tensor: [{tensor.min():.3f}, {tensor.max():.3f}]")
            
            print(f"  ✓ Image validation passed")
            
        except Exception as e:
            print(f"  ERROR: Failed to process image: {e}")
            issues_found += 1
    
    print(f"\nValidation complete!")
    print(f"Issues found: {issues_found}/{min(max_images, len(image_files))}")
    
    return issues_found == 0

def main():
    parser = argparse.ArgumentParser(description="Validate image dataset for training")
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument("--max_images", type=int, default=10, 
                       help="Maximum number of images to validate (default: 10)")
    
    args = parser.parse_args()
    
    success = validate_image_dataset(args.dataset_path, args.max_images)
    if not success:
        print("\n⚠️  Issues found in dataset. Please fix them before training.")
        exit(1)
    else:
        print("\n✅ Dataset validation passed!")
        exit(0)

if __name__ == "__main__":
    main()
