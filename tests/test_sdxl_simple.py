#!/usr/bin/env python3
"""
Simple SDXL Pipeline Test Script

This script helps diagnose SDXL pipeline issues independently of the training code.
Run this to test if your SDXL setup works before attempting training.
"""

import torch
import numpy as np
from pathlib import Path
from diffusers import StableDiffusionXLPipeline
import warnings

def test_basic_setup():
    """Test basic PyTorch and GPU setup."""
    print("=== Basic Setup Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

def test_model_loading(model_name="stabilityai/stable-diffusion-xl-base-1.0"):
    """Test loading the SDXL model."""
    print(f"=== Model Loading Test: {model_name} ===")
    
    try:
        # Test different loading configurations
        configs = [
            {"dtype": torch.float32, "variant": None, "desc": "float32"},
            {"dtype": torch.float16, "variant": "fp16", "desc": "fp16"},
        ]
        
        for config in configs:
            try:
                print(f"Testing {config['desc']}...")
                
                kwargs = {
                    "torch_dtype": config["dtype"],
                    "safety_checker": None,
                    "requires_safety_checker": False,
                    "use_safetensors": True,
                }
                
                if config["variant"]:
                    kwargs["variant"] = config["variant"]
                
                pipeline = StableDiffusionXLPipeline.from_pretrained(model_name, **kwargs)
                
                if torch.cuda.is_available():
                    pipeline = pipeline.to("cuda")
                
                print(f"  ✓ Model loaded successfully with {config['desc']}")
                print(f"  UNet device: {pipeline.unet.device}")
                print(f"  UNet dtype: {pipeline.unet.dtype}")
                print(f"  VAE device: {pipeline.vae.device}")
                print(f"  VAE dtype: {pipeline.vae.dtype}")
                
                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return True
                
            except Exception as e:
                print(f"  ✗ Failed to load with {config['desc']}: {e}")
                continue
        
        print("  ✗ All loading configurations failed")
        return False
        
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return False

def test_image_generation(model_name="stabilityai/stable-diffusion-xl-base-1.0"):
    """Test image generation with the SDXL model."""
    print(f"=== Image Generation Test: {model_name} ===")
    
    try:
        # Load model with most compatible settings
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Most compatible
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
        )
        
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
        
        # Simple prompt
        prompt = "a red apple on a white table"
        
        print(f"Generating image with prompt: '{prompt}'")
        
        # Generate with conservative settings
        with torch.no_grad():
            result = pipeline(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512,
                generator=torch.Generator(device=pipeline.device).manual_seed(42) if torch.cuda.is_available() else torch.Generator().manual_seed(42)
            )
        
        image = result.images[0]
        
        # Analyze image
        img_array = np.array(image)
        
        print("Image statistics:")
        print(f"  Shape: {img_array.shape}")
        print(f"  Dtype: {img_array.dtype}")
        print(f"  Min: {img_array.min()}")
        print(f"  Max: {img_array.max()}")
        print(f"  Mean: {img_array.mean():.2f}")
        print(f"  Std: {img_array.std():.2f}")
        print(f"  Has NaN: {np.isnan(img_array).any()}")
        print(f"  Has Inf: {np.isinf(img_array).any()}")
        
        # Save image
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        image_path = output_dir / "sdxl_test.png"
        image.save(image_path)
        print(f"  Image saved to: {image_path}")
        
        # Evaluate quality
        if img_array.mean() > 10 and img_array.std() > 5:
            print("  ✓ Image generation successful!")
            return True
        else:
            print(f"  ✗ Poor image quality - likely black/corrupted image")
            print(f"    Mean brightness: {img_array.mean():.2f} (should be > 10)")
            print(f"    Standard deviation: {img_array.std():.2f} (should be > 5)")
            return False
            
    except Exception as e:
        print(f"  ✗ Image generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'pipeline' in locals():
            del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Run all tests."""
    print("SDXL Pipeline Diagnostic Tool")
    print("=" * 50)
    
    # Suppress some warnings for cleaner output
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Run tests
    setup_ok = test_basic_setup()
    
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    print(f"Using model: {model_name}")
    print()
    
    loading_ok = test_model_loading(model_name)
    
    if loading_ok:
        generation_ok = test_image_generation(model_name)
        
        if generation_ok:
            print("\n" + "=" * 50)
            print("✓ ALL TESTS PASSED!")
            print("Your SDXL setup appears to be working correctly.")
            print("You can proceed with the DoRA training.")
        else:
            print("\n" + "=" * 50)
            print("✗ IMAGE GENERATION FAILED")
            print("The model loads but produces invalid images.")
            print("This suggests an environment or model issue.")
            print("\nSuggestions:")
            print("1. Update diffusers: pip install --upgrade diffusers")
            print("2. Try a different model")
            print("3. Check GPU memory and drivers")
            print("4. Try running on CPU (add --device cpu argument)")
    else:
        print("\n" + "=" * 50)
        print("✗ MODEL LOADING FAILED")
        print("Cannot load the SDXL model.")
        print("\nSuggestions:")
        print("1. Check internet connection")
        print("2. Update transformers: pip install --upgrade transformers")
        print("3. Try downloading the model manually")
        print("4. Check available disk space")

if __name__ == "__main__":
    main()
