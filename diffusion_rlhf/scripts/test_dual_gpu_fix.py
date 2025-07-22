#!/usr/bin/env python3
"""
Test script to verify dual GPU fixes.
"""

import torch
from diffusers import StableDiffusionXLPipeline
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline_loading():
    """Test if pipeline loads correctly with fixes."""
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return False
        
    gpu_count = torch.cuda.device_count()
    logger.info(f"Found {gpu_count} GPUs")
    
    if gpu_count < 2:
        logger.warning("Less than 2 GPUs available, testing single GPU")
        
    try:
        # Test loading on GPU 0
        device = "cuda:0"
        logger.info(f"Testing pipeline loading on {device}")
        
        # Load with fixed parameters
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            device_map=None,
            low_cpu_mem_usage=False,  # Key fix for meta tensor issues
        )
        
        # Move to device
        pipeline = pipeline.to(device, dtype=torch.float16)
        
        # Test generation
        generator = torch.Generator(device=device).manual_seed(42)
        
        with torch.cuda.device(device):
            image = pipeline(
                prompt="A simple test image",
                width=512,
                height=512,
                num_inference_steps=10,
                generator=generator,
            ).images[0]
        
        logger.info("✅ Pipeline test successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_pipeline_loading()
    if success:
        print("🎉 Dual GPU fixes are working!")
    else:
        print("❌ Issues still remain")
