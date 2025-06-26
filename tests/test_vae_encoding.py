#!/usr/bin/env python3
"""
VAE encoding test script to diagnose NaN issues.
"""

import torch
from PIL import Image
from pathlib import Path
from diffusers import AutoencoderKL
from torchvision import transforms
import argparse

def test_vae_with_dataset(dataset_path: str, model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"):
    """Test VAE encoding with actual dataset images."""
    
    print(f"Testing VAE with dataset: {dataset_path}")
    print(f"Using model: {model_name}")
    
    # Load VAE in float32
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        model_name,
        subfolder="vae",
        torch_dtype=torch.float32
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae.to(device)
    vae.eval()
    
    print(f"VAE loaded on {device}")
    
    # Setup transforms (same as training)
    transforms_list = transforms.Compose([
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Get image files
    dataset_path = Path(dataset_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(dataset_path.rglob(f'*{ext}')))
        image_files.extend(list(dataset_path.rglob(f'*{ext.upper()}')))
    
    if not image_files:
        print("ERROR: No image files found")
        return False
    
    print(f"Found {len(image_files)} images")
    
    # Test first 10 images
    success_count = 0
    fail_count = 0
    
    for i, image_path in enumerate(image_files[:10]):
        print(f"\nTesting image {i+1}: {image_path.name}")
        
        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            tensor = transforms_list(image).unsqueeze(0).to(device)
            
            print(f"  Image tensor: shape={tensor.shape}, dtype={tensor.dtype}")
            print(f"  Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
            
            # Test VAE encoding
            with torch.no_grad():
                # Test with float32
                tensor_f32 = tensor.float()
                latents = vae.encode(tensor_f32).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                if torch.isnan(latents).any():
                    print(f"  ❌ NaN detected in latents!")
                    print(f"     NaN count: {torch.isnan(latents).sum().item()}")
                    fail_count += 1
                elif torch.isinf(latents).any():
                    print(f"  ❌ Inf detected in latents!")
                    print(f"     Inf count: {torch.isinf(latents).sum().item()}")
                    fail_count += 1
                else:
                    print(f"  ✅ Success! Latents: shape={latents.shape}, range=[{latents.min():.3f}, {latents.max():.3f}]")
                    success_count += 1
                    
        except Exception as e:
            print(f"  ❌ Error: {e}")
            fail_count += 1
    
    print(f"\nResults: {success_count} success, {fail_count} failures")
    
    if fail_count > 0:
        print("\n🔍 Debugging suggestions:")
        print("1. Check if images have extreme values or corruption")
        print("2. Try with different mixed precision settings")
        print("3. Check GPU memory and drivers")
        print("4. Try with smaller batch size")
    
    return fail_count == 0

def main():
    parser = argparse.ArgumentParser(description="Test VAE encoding with dataset")
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument("--model_name", default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="SDXL model name")
    
    args = parser.parse_args()
    
    success = test_vae_with_dataset(args.dataset_path, args.model_name)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
