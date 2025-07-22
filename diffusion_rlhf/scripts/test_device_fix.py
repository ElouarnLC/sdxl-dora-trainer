#!/usr/bin/env python3
"""
Quick test to verify device placement is working correctly.
"""

import torch
from generate_images import setup_pipeline

def test_device_placement():
    """Test that all pipeline components are on the correct device."""
    print("Testing device placement fix...")
    
    # Test with CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    # Set up pipeline
    pipeline = setup_pipeline(
        model_name="stabilityai/stable-diffusion-xl-base-1.0",
        device=device
    )
    
    # Check device placement
    print(f"Pipeline device: {getattr(pipeline, '_device', 'unknown')}")
    print(f"UNet device: {next(pipeline.unet.parameters()).device}")
    print(f"VAE device: {next(pipeline.vae.parameters()).device}")
    print(f"Text encoder device: {next(pipeline.text_encoder.parameters()).device}")
    
    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2:
        print(f"Text encoder 2 device: {next(pipeline.text_encoder_2.parameters()).device}")
    
    # Try a simple generation
    print("\nTesting generation...")
    generator = torch.Generator(device=device).manual_seed(42)
    
    try:
        image = pipeline(
            prompt="A beautiful sunset",
            width=512,
            height=512,
            num_inference_steps=20,
            generator=generator,
        ).images[0]
        print("✅ Generation successful!")
        return True
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_device_placement()
    if success:
        print("\n🎉 Device placement fix is working correctly!")
    else:
        print("\n💥 Device placement fix needs more work.")
