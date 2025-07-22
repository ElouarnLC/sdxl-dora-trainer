#!/usr/bin/env python3
"""
SDXL image generation script for preference dataset creation.

Generates 2 images per prompt (with seeds 0 and 1) using SDXL base model.
Saves images and metadata for downstream preference annotation and training.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from peft import PeftModel
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_pipeline(
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    dora_weights_path: str = None,
) -> StableDiffusionXLPipeline:
    """
    Set up SDXL pipeline for image generation.
    
    Parameters
    ----------
    model_name : str
        Hugging Face model name
    device : str
        Device to use ("auto", "cuda", "cpu")
    torch_dtype : torch.dtype
        Torch dtype for model
    dora_weights_path : str, optional
        Path to DoRA weights directory
        
    Returns
    -------
    StableDiffusionXLPipeline
        Configured pipeline
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading SDXL pipeline on {device}")
    
    # Load pipeline with device placement
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        device_map=None,  # Don't use automatic device mapping
    )
    
    if device == "cuda":
        # Move entire pipeline to CUDA first
        pipeline = pipeline.to(device)
        
        # Enable memory optimizations for CUDA (but be careful with cpu_offload)
        pipeline.enable_vae_slicing()
        pipeline.enable_attention_slicing()
        
        # Only enable CPU offload if we have memory issues
        # pipeline.enable_model_cpu_offload()  # Comment out for now
    else:
        # For CPU, move to device
        pipeline = pipeline.to(device)
    
    # Load DoRA weights if provided
    if dora_weights_path:
        logger.info(f"Loading DoRA weights from {dora_weights_path}")
        
        # Check if DoRA weights exist
        weights_path = Path(dora_weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"DoRA weights path does not exist: {dora_weights_path}")
        
        # Check for required files
        adapter_config = weights_path / "adapter_config.json"
        if not adapter_config.exists():
            raise FileNotFoundError(f"adapter_config.json not found in {dora_weights_path}")
        
        try:
            # Load DoRA weights to UNet with proper precision handling
            # Important: Use fp32 for DoRA weights to avoid black image issue
            # See BLACK_IMAGE_FIX.md - fp16 can cause NaN values in DoRA weights
            pipeline.unet = PeftModel.from_pretrained(
                pipeline.unet,
                dora_weights_path,
                torch_dtype=torch.float32  # Use fp32 for stable DoRA loading
            )
            # Move to device and convert to fp16 only after loading (if using fp16)
            if torch_dtype == torch.float16:
                pipeline.unet = pipeline.unet.to(device, dtype=torch.float16)
                logger.info("DoRA weights loaded (fp32→fp16 conversion)")
            else:
                pipeline.unet = pipeline.unet.to(device)
                logger.info("DoRA weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DoRA weights: {e}")
            logger.error("Tip: Check BLACK_IMAGE_FIX.md for DoRA troubleshooting")
            raise
    else:
        logger.info("No DoRA weights specified, using base model only")
    
    # Final device check - ensure all components are on the same device
    logger.info(f"Final device check - ensuring all components are on {device}")
    pipeline = pipeline.to(device)
    
    # Double-check critical components
    if hasattr(pipeline, 'unet') and pipeline.unet is not None:
        pipeline.unet = pipeline.unet.to(device)
    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
        pipeline.vae = pipeline.vae.to(device)
    if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
        pipeline.text_encoder = pipeline.text_encoder.to(device)
    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
        pipeline.text_encoder_2 = pipeline.text_encoder_2.to(device)
    
    return pipeline


def generate_images(
    pipeline: StableDiffusionXLPipeline,
    prompt: str,
    prompt_id: int,
    output_dir: Path,
    seeds: list[int] = None,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> list[dict[str, Any]]:
    """
    Generate images for a single prompt with multiple seeds.
    
    Parameters
    ----------
    pipeline : StableDiffusionXLPipeline
        SDXL pipeline
    prompt : str
        Text prompt
    prompt_id : int
        Unique prompt identifier
    output_dir : Path
        Output directory
    seeds : list[int]
        Random seeds to use
    width : int
        Image width
    height : int
        Image height
    num_inference_steps : int
        Number of inference steps
    guidance_scale : float
        Classifier-free guidance scale
        
    Returns
    -------
    list[dict[str, Any]]
        List of generated image metadata
    """
    if seeds is None:
        seeds = [0, 1]
        
    results = []
    
    # Create prompt-specific directory
    prompt_dir = output_dir / str(prompt_id)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    for seed in seeds:
        # Ensure generator is on the correct device
        if hasattr(pipeline, 'device'):
            device = pipeline.device
        else:
            device = pipeline.unet.device
        generator = torch.Generator(device=device).manual_seed(seed)
        
        logger.info(f"Generating image for prompt {prompt_id}, seed {seed}")
        
        # Generate image
        try:
            image = pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            
            # Clear CUDA cache if using CUDA to prevent memory fragmentation
            if torch.cuda.is_available() and 'cuda' in str(device):
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "device" in str(e).lower():
                logger.error(
                    f"Device/memory error for prompt {prompt_id}, seed {seed}: {e}"
                )
                # Clear cache and retry once
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            else:
                raise
        
        # Save image
        image_path = prompt_dir / f"{seed}.png"
        image.save(image_path)
        
        # Store metadata
        results.append({
            "prompt_id": prompt_id,
            "seed": seed,
            "path": str(image_path),
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        })
    
    return results


def main() -> None:
    """Main generation function."""
    parser = argparse.ArgumentParser(
        description="Generate images from prompts using SDXL"
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
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
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (number of prompts to process in parallel)",
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
    pipeline = setup_pipeline(
        model_name=args.model,
        device=args.device,
        dora_weights_path=args.dora_weights,
    )
    
    # Generate images
    all_results = []
    
    for _, row in tqdm(
        prompts_df.iterrows(),
        total=len(prompts_df),
        desc="Generating images"
    ):
        prompt_id = row["prompt_id"]
        prompt = row["prompt"]
        
        try:
            results = generate_images(
                pipeline=pipeline,
                prompt=prompt,
                prompt_id=prompt_id,
                output_dir=output_dir,
                seeds=args.seeds,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
            )
            all_results.extend(results)
            
        except Exception as e:
            logger.error(f"Failed to generate images for prompt {prompt_id}: {e}")
            continue
    
    # Save metadata
    results_df = pd.DataFrame(all_results)
    results_file = output_dir.parent / "generated.csv"
    results_df.to_csv(results_file, index=False)
    
    logger.info(f"Generated {len(all_results)} images")
    logger.info(f"Metadata saved to {results_file}")
    logger.info(f"Images saved to {output_dir}")


if __name__ == "__main__":
    main()
