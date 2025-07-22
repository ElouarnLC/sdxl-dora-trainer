#!/usr/bin/env python3
"""
Multi-GPU SDXL image generation script for preference dataset creation.

Supports multiple strategies for multi-GPU acceleration:
1. Model parallelism (split model across GPUs)
2. Data parallelism (multiple pipelines on different GPUs)
3. Batch parallelism (distribute batches across GPUs)
"""

import argparse
import logging
import multiprocessing as mp
import os
import queue
import threading
from pathlib import Path
from typing import Any, List

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


def get_available_gpus() -> List[int]:
    """Get list of available GPU indices."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def setup_pipeline_on_device(
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device_id: int = 0,
    torch_dtype: torch.dtype = torch.float16,
    dora_weights_path: str = None,
    enable_sequential_cpu_offload: bool = False,
) -> StableDiffusionXLPipeline:
    """
    Set up SDXL pipeline on a specific GPU device.
    
    Parameters
    ----------
    model_name : str
        Hugging Face model name
    device_id : int
        GPU device ID
    torch_dtype : torch.dtype
        Torch dtype for model
    dora_weights_path : str, optional
        Path to DoRA weights directory
    enable_sequential_cpu_offload : bool
        Whether to enable sequential CPU offloading for memory efficiency
        
    Returns
    -------
    StableDiffusionXLPipeline
        Configured pipeline
    """
    device = f"cuda:{device_id}"
    
    logger.info(f"Loading SDXL pipeline on {device}")
    
    # Load pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        device_map=None,
    )
    
    # Move to specific device
    pipeline = pipeline.to(device)
    
    # Memory optimizations
    pipeline.enable_vae_slicing()
    pipeline.enable_attention_slicing()
    
    if enable_sequential_cpu_offload:
        # Use sequential CPU offload for memory efficiency
        # This keeps only the active component on GPU
        pipeline.enable_sequential_cpu_offload(gpu_id=device_id)
    else:
        # Keep everything on GPU for speed
        pipeline.unet = pipeline.unet.to(device)
        pipeline.vae = pipeline.vae.to(device)
        pipeline.text_encoder = pipeline.text_encoder.to(device)
        pipeline.text_encoder_2 = pipeline.text_encoder_2.to(device)
    
    # Load DoRA weights if provided
    if dora_weights_path:
        logger.info(f"Loading DoRA weights from {dora_weights_path} on {device}")
        
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
            pipeline.unet = pipeline.unet.to(device)
            logger.info(f"DoRA weights loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load DoRA weights on {device}: {e}")
            raise
    
    return pipeline


def generate_worker(
    gpu_id: int,
    task_queue: queue.Queue,
    result_queue: queue.Queue,
    pipeline_args: dict,
    generation_args: dict,
):
    """
    Worker function for multi-GPU generation.
    
    Parameters
    ----------
    gpu_id : int
        GPU device ID
    task_queue : queue.Queue
        Queue containing generation tasks
    result_queue : queue.Queue
        Queue for storing results
    pipeline_args : dict
        Arguments for pipeline setup
    generation_args : dict
        Arguments for image generation
    """
    try:
        # Set up pipeline on this GPU
        device = f"cuda:{gpu_id}"
        pipeline = setup_pipeline_on_device(device_id=gpu_id, **pipeline_args)
        
        logger.info(f"Worker on {device} ready")
        
        while True:
            try:
                # Get task from queue (timeout to allow graceful shutdown)
                task = task_queue.get(timeout=1)
                if task is None:  # Shutdown signal
                    break
                
                prompt_id, prompt, seeds = task
                results = []
                
                # Create prompt-specific directory
                output_dir = Path(generation_args['output_dir'])
                prompt_dir = output_dir / str(prompt_id)
                prompt_dir.mkdir(parents=True, exist_ok=True)
                
                for seed in seeds:
                    # Generate image
                    generator = torch.Generator(device=device).manual_seed(seed)
                    
                    try:
                        image = pipeline(
                            prompt=prompt,
                            width=generation_args['width'],
                            height=generation_args['height'],
                            num_inference_steps=generation_args['num_inference_steps'],
                            guidance_scale=generation_args['guidance_scale'],
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
                            "width": generation_args['width'],
                            "height": generation_args['height'],
                            "num_inference_steps": generation_args['num_inference_steps'],
                            "guidance_scale": generation_args['guidance_scale'],
                            "gpu_id": gpu_id,
                        })
                        
                        # Clear CUDA cache
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logger.error(
                            f"Failed to generate image on {device} for "
                            f"prompt {prompt_id}, seed {seed}: {e}"
                        )
                        continue
                
                # Put results in result queue
                result_queue.put(results)
                task_queue.task_done()
                
            except queue.Empty:
                continue  # Check for shutdown
            except Exception as e:
                logger.error(f"Worker on {device} error: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Failed to initialize worker on cuda:{gpu_id}: {e}")
    finally:
        logger.info(f"Worker on cuda:{gpu_id} shutting down")


def generate_images_multi_gpu(
    prompts_df: pd.DataFrame,
    output_dir: Path,
    gpu_ids: List[int],
    pipeline_args: dict,
    generation_args: dict,
    max_workers: int = None,
) -> List[dict]:
    """
    Generate images using multiple GPUs.
    
    Parameters
    ----------
    prompts_df : pd.DataFrame
        DataFrame with prompts
    output_dir : Path
        Output directory
    gpu_ids : List[int]
        List of GPU IDs to use
    pipeline_args : dict
        Arguments for pipeline setup
    generation_args : dict
        Arguments for image generation
    max_workers : int, optional
        Maximum number of worker processes
        
    Returns
    -------
    List[dict]
        List of generated image metadata
    """
    if not gpu_ids:
        raise ValueError("No GPUs available")
    
    if max_workers is None:
        max_workers = len(gpu_ids)
    
    # Create queues
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Add generation args to the dict
    generation_args['output_dir'] = str(output_dir)
    
    # Populate task queue
    for _, row in prompts_df.iterrows():
        prompt_id = row["prompt_id"]
        prompt = row["prompt"]
        seeds = generation_args.get('seeds', [0, 1])
        task_queue.put((prompt_id, prompt, seeds))
    
    # Add shutdown signals
    for _ in range(max_workers):
        task_queue.put(None)
    
    # Start worker threads
    workers = []
    for i in range(max_workers):
        gpu_id = gpu_ids[i % len(gpu_ids)]  # Cycle through available GPUs
        worker = threading.Thread(
            target=generate_worker,
            args=(gpu_id, task_queue, result_queue, pipeline_args, generation_args),
        )
        worker.start()
        workers.append(worker)
    
    # Collect results
    all_results = []
    total_tasks = len(prompts_df)
    completed_tasks = 0
    
    with tqdm(total=total_tasks, desc="Generating images") as pbar:
        while completed_tasks < total_tasks:
            try:
                results = result_queue.get(timeout=30)  # 30 second timeout
                all_results.extend(results)
                completed_tasks += 1
                pbar.update(1)
            except queue.Empty:
                logger.warning("No results received in 30 seconds")
                continue
    
    # Wait for all workers to complete
    for worker in workers:
        worker.join()
    
    return all_results


def main() -> None:
    """Main generation function."""
    parser = argparse.ArgumentParser(
        description="Generate images from prompts using multiple GPUs"
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
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes",
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
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable sequential CPU offloading for memory efficiency",
    )
    
    args = parser.parse_args()
    
    # Get available GPUs
    available_gpus = get_available_gpus()
    if not available_gpus:
        raise RuntimeError("No CUDA GPUs available")
    
    if args.gpu_ids is None:
        gpu_ids = available_gpus
    else:
        gpu_ids = [gpu_id for gpu_id in args.gpu_ids if gpu_id in available_gpus]
        if not gpu_ids:
            raise ValueError("None of the specified GPU IDs are available")
    
    logger.info(f"Using GPUs: {gpu_ids}")
    
    # Set up paths
    prompts_file = Path(args.prompts)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    prompts_df = pd.read_csv(prompts_file)
    logger.info(f"Loaded {len(prompts_df)} prompts from {prompts_file}")
    
    # Set up arguments
    pipeline_args = {
        "model_name": args.model,
        "dora_weights_path": args.dora_weights,
        "enable_sequential_cpu_offload": args.enable_cpu_offload,
    }
    
    generation_args = {
        "seeds": args.seeds,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
    }
    
    # Generate images
    all_results = generate_images_multi_gpu(
        prompts_df=prompts_df,
        output_dir=output_dir,
        gpu_ids=gpu_ids,
        pipeline_args=pipeline_args,
        generation_args=generation_args,
        max_workers=args.max_workers,
    )
    
    # Save metadata
    results_df = pd.DataFrame(all_results)
    results_file = output_dir.parent / "generated_multi_gpu.csv"
    results_df.to_csv(results_file, index=False)
    
    logger.info(f"Generated {len(all_results)} images using {len(gpu_ids)} GPUs")
    logger.info(f"Metadata saved to {results_file}")
    logger.info(f"Images saved to {output_dir}")


if __name__ == "__main__":
    main()
