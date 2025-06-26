#!/usr/bin/env python3
"""
Quick test script to verify device allocation and tensor shapes.
"""

import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
import sys
import os

def test_device_allocation():
    """Test that all models are properly allocated to GPU."""
    
    print("Testing device allocation...")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, testing on CPU")
        device = torch.device("cpu")
    else:
        print(f"CUDA available, testing on GPU")
        device = torch.device("cuda")
    
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    
    try:
        # Load tokenizer
        tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        
        # Load text encoders
        text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=torch.float16
        )
        text_encoder_2 = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder_2", torch_dtype=torch.float16
        )
        
        # Move to device
        text_encoder = text_encoder.to(device)
        text_encoder_2 = text_encoder_2.to(device)
        
        # Test tokenization and encoding
        test_prompt = "a beautiful landscape"
        inputs = tokenizer(
            test_prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs.input_ids.to(device)
        print(f"Input IDs shape: {input_ids.shape}, device: {input_ids.device}")
        
        # Test text encoders
        with torch.no_grad():
            output_1 = text_encoder(input_ids)
            output_2 = text_encoder_2(input_ids)
            
            embeds_1 = output_1.last_hidden_state
            embeds_2 = output_2.last_hidden_state
            pooled_embeds = output_2.pooler_output
            
            print(f"Text encoder 1 output shape: {embeds_1.shape}, device: {embeds_1.device}")
            print(f"Text encoder 2 output shape: {embeds_2.shape}, device: {embeds_2.device}")
            print(f"Pooled embeddings shape: {pooled_embeds.shape}, device: {pooled_embeds.device}")
            
            # Concatenate embeddings
            combined_embeds = torch.cat([embeds_1, embeds_2], dim=-1)
            print(f"Combined embeddings shape: {combined_embeds.shape}, device: {combined_embeds.device}")
        
        print("✓ Device allocation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Device allocation test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_device_allocation()
    sys.exit(0 if success else 1)
