#!/usr/bin/env python3
"""
Simple multi-GPU SDXL generation using DataParallel approach.

This script provides an easier way to use multiple GPUs by leveraging
PyTorch's DataParallel functionality for faster image generation.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline
from peft import PeftModel
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MultiGPUPipelineWrapper(nn.Module):
    """Wrapper to make SDXL UNet compatible with DataParallel."""
    
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    
    def forward(self, *args, **kwargs):
        return self.unet(*args, **kwargs)


def setup_multi_gpu_pipeline(
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype: torch.dtype = torch.float16,
    dora_weights_path: Optional[str] = None,
    gpu_ids: Optional[list[int]] = None,
) -> StableDiffusionXLPipeline:
    """
    Set up SDXL pipeline with multi-GPU support.
    
    Parameters
    ----------
    model_name : str
        Hugging Face model name
    torch_dtype : torch.dtype
        Torch dtype for model
    dora_weights_path : str, optional
        Path to DoRA weights directory
    gpu_ids : list[int], optional
        List of GPU IDs to use (default: all available)
        
    Returns
    -------
    StableDiffusionXLPipeline
        Configured pipeline with multi-GPU support
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices found")
    
    if gpu_ids is None:
        gpu_ids = list(range(device_count))
    
    logger.info(f"Setting up pipeline on GPUs: {gpu_ids}")
    
    # Load pipeline on primary GPU
    primary_device = f"cuda:{gpu_ids[0]}"
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        device_map=None,
    )
    
    # Move to primary device
    pipeline = pipeline.to(primary_device)
    
    # Load DoRA weights if provided
    if dora_weights_path:
        logger.info(f"Loading DoRA weights from {dora_weights_path}")
        
        weights_path = Path(dora_weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"DoRA weights path does not exist: {dora_weights_path}"
            )
        
        adapter_config = weights_path / "adapter_config.json"
        if not adapter_config.exists():
            raise FileNotFoundError(
                f"adapter_config.json not found in {dora_weights_path}"
            )
        
        try:
            pipeline.unet = PeftModel.from_pretrained(
                pipeline.unet,
                dora_weights_path,
                torch_dtype=torch_dtype
            )
            pipeline.unet = pipeline.unet.to(primary_device)
            logger.info("DoRA weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DoRA weights: {e}")
            raise
    
    # Set up multi-GPU for UNet (the most compute-intensive part)
    if len(gpu_ids) > 1:
        logger.info(f"Setting up DataParallel on GPUs: {gpu_ids}")
        
        # Wrap UNet for DataParallel
        unet_wrapper = MultiGPUPipelineWrapper(pipeline.unet)
        unet_parallel = nn.DataParallel(unet_wrapper, device_ids=gpu_ids)
        
        # Replace the original UNet with parallel version
        pipeline.unet = unet_parallel.module.unet
        pipeline._unet_parallel = unet_parallel
        
        # Keep other components on primary GPU
        pipeline.vae = pipeline.vae.to(primary_device)
        pipeline.text_encoder = pipeline.text_encoder.to(primary_device)
        pipeline.text_encoder_2 = pipeline.text_encoder_2.to(primary_device)
        
        logger.info("Multi-GPU setup complete")
    else:
        logger.info("Single GPU setup complete")
    
    # Enable memory optimizations
    pipeline.enable_vae_slicing()
    pipeline.enable_attention_slicing()
    
    return pipeline


def generate_images_batch(
    pipeline: StableDiffusionXLPipeline,
    prompts_batch: list[str],
    batch_size: int = 1,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seeds: Optional[list[int]] = None,
) -> list[tuple]:
    """
    Generate images for a batch of prompts.
    
    Parameters
    ----------
    pipeline : StableDiffusionXLPipeline
        SDXL pipeline
    prompts_batch : list[str]
        List of prompts to generate
    batch_size : int
        Batch size for generation
    width : int
        Image width
    height : int
        Image height
    num_inference_steps : int
        Number of inference steps
    guidance_scale : float
        Classifier-free guidance scale
    seeds : list[int], optional
        Random seeds (default: [0, 1])
        
    Returns
    -------
    list[tuple]
        List of (prompt, seed, image) tuples
    """
    if seeds is None:
        seeds = [0, 1]
    
    results = []
    device = next(pipeline.unet.parameters()).device
    
    for prompt in prompts_batch:
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            
            try:
                # Generate with increased batch size for multi-GPU efficiency
                images = pipeline(
                    prompt=[prompt] * batch_size,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=[generator] * batch_size,
                ).images
                
                # Take the first image (since we're repeating the same prompt)
                results.append((prompt, seed, images[0]))
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Failed to generate image for prompt '{prompt}', seed {seed}: {e}")
                continue
    
    return results


def main() -> None:
    """Main generation function."""
    parser = argparse.ArgumentParser(
        description="Generate images using multi-GPU SDXL pipeline"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to prompts CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/images",
        help="Output directory for images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="SDXL model name",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help="GPU IDs to use (default: all available)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Random seeds to use",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--dora-weights",
        type=str,
        default=None,
        help="Path to DoRA weights directory (optional)",
    )
    
    args = parser.parse_args()
    
    # Set up paths
    prompts_file = Path(args.prompts)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    prompts_df = pd.read_csv(prompts_file)
    logger.info(f"Loaded {len(prompts_df)} prompts from {prompts_file}")
    
    # Set up pipeline
    pipeline = setup_multi_gpu_pipeline(
        model_name=args.model,
        dora_weights_path=args.dora_weights,
        gpu_ids=args.gpu_ids,
    )
    
    # Generate images
    all_results = []
    
    for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df), desc="Generating images"):
        prompt_id = row["prompt_id"]
        prompt = row["prompt"]
        
        try:
            # Generate images for this prompt
            results = generate_images_batch(
                pipeline=pipeline,
                prompts_batch=[prompt],
                batch_size=args.batch_size,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seeds=args.seeds,
            )
            
            # Save images and collect metadata
            prompt_dir = output_dir / str(prompt_id)
            prompt_dir.mkdir(parents=True, exist_ok=True)
            
            for prompt_text, seed, image in results:
                # Save image
                image_path = prompt_dir / f"{seed}.png"
                image.save(image_path)
                
                # Store metadata
                all_results.append({
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "path": str(image_path),
                    "prompt": prompt_text,
                    "width": args.width,
                    "height": args.height,
                    "num_inference_steps": args.steps,
                    "guidance_scale": args.guidance_scale,
                })
            
        except Exception as e:
            logger.error(f"Failed to generate images for prompt {prompt_id}: {e}")
            continue
    
    # Save metadata
    results_df = pd.DataFrame(all_results)
    results_file = output_dir.parent / "generated_dataparallel.csv"
    results_df.to_csv(results_file, index=False)
    
    logger.info(f"Generated {len(all_results)} images")
    logger.info(f"Metadata saved to {results_file}")
    logger.info(f"Images saved to {output_dir}")


if __name__ == "__main__":
    main()
