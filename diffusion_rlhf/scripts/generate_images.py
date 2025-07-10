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
        
    Returns
    -------
    StableDiffusionXLPipeline
        Configured pipeline
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading SDXL pipeline on {device}")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        device_map=device,
    )
    
    # Enable memory optimizations
    if device == "cuda":
        pipeline.enable_model_cpu_offload()
        pipeline.enable_vae_slicing()
        pipeline.enable_attention_slicing()
    
    return pipeline


def generate_images(
    pipeline: StableDiffusionXLPipeline,
    prompt: str,
    prompt_id: int,
    output_dir: Path,
    seeds: list[int] = [0, 1],
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
    results = []
    
    # Create prompt-specific directory
    prompt_dir = output_dir / str(prompt_id)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    for seed in seeds:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        
        logger.info(f"Generating image for prompt {prompt_id}, seed {seed}")
        
        # Generate image
        image = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
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
    )
    
    # Generate images
    all_results = []
    
    for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df), desc="Generating images"):
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
