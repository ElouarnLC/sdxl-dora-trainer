#!/usr/bin/env python3
"""
Dual-GPU SDXL generation script using two separate pipelines.
Each GPU runs its own complete pipeline for maximum throughput.
"""

import argparse
import concurrent.futures
import logging
import queue
import threading
from pathlib import Path

import pandas as pd
import torch
from diffusers import StableDiffusionXLPipeline
from peft import PeftModel
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_pipeline_on_gpu(
    gpu_id: int,
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
    dora_weights_path: str | None = None,
) -> StableDiffusionXLPipeline:
    """Set up SDXL pipeline on a specific GPU."""
    device = f"cuda:{gpu_id}"
    
    logger.info(f"Loading pipeline on {device}")
    
    # Load pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
        device_map=None,
    )
    
    # Move to specific GPU
    pipeline = pipeline.to(device)
    
    # Load DoRA weights if provided
    if dora_weights_path:
        logger.info(f"Loading DoRA weights on {device}")
        try:
            pipeline.unet = PeftModel.from_pretrained(
                pipeline.unet,
                dora_weights_path,
                torch_dtype=torch.float16
            )
            pipeline.unet = pipeline.unet.to(device)
            logger.info(f"DoRA weights loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load DoRA weights on {device}: {e}")
            raise
    
    # Memory optimizations
    pipeline.enable_vae_slicing()
    pipeline.enable_attention_slicing()
    
    # Ensure all components are on the correct device
    pipeline = pipeline.to(device)
    
    return pipeline


def gpu_worker(
    gpu_id: int,
    task_queue: queue.Queue,
    result_queue: queue.Queue,
    model_name: str,
    dora_weights_path: str | None,
    generation_args: dict,
):
    """Worker function for a single GPU."""
    try:
        # Set up pipeline on this GPU
        pipeline = setup_pipeline_on_gpu(gpu_id, model_name, dora_weights_path)
        device = f"cuda:{gpu_id}"
        
        logger.info(f"GPU {gpu_id} worker ready")
        
        while True:
            try:
                # Get task from queue
                task = task_queue.get(timeout=1)
                if task is None:  # Shutdown signal
                    break
                
                prompt_id, prompt, seed, output_dir = task
                
                # Generate image
                generator = torch.Generator(device=device).manual_seed(seed)
                
                image = pipeline(
                    prompt=prompt,
                    width=generation_args['width'],
                    height=generation_args['height'],
                    num_inference_steps=generation_args['steps'],
                    guidance_scale=generation_args.get('guidance_scale', 7.5),
                    generator=generator,
                ).images[0]
                
                # Save image
                prompt_dir = Path(output_dir) / str(prompt_id)
                prompt_dir.mkdir(parents=True, exist_ok=True)
                image_path = prompt_dir / f"{seed}.png"
                image.save(image_path)
                
                # Store result
                result = {
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "path": str(image_path),
                    "prompt": prompt,
                    "gpu_id": gpu_id,
                }
                result_queue.put(result)
                
                # Clear cache
                torch.cuda.empty_cache()
                
                task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error on GPU {gpu_id}: {e}")
                task_queue.task_done()
                continue
                
    except Exception as e:
        logger.error(f"Failed to initialize GPU {gpu_id} worker: {e}")
    finally:
        logger.info(f"GPU {gpu_id} worker shutting down")


def main():
    """Main function using dual GPU pipelines."""
    parser = argparse.ArgumentParser(description="Dual-GPU SDXL image generation")
    parser.add_argument("--prompts", required=True, help="Path to prompts CSV")
    parser.add_argument("--output", default="data/images", help="Output directory")
    parser.add_argument(
        "--model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Model name"
    )
    parser.add_argument("--dora-weights", help="DoRA weights path (optional)")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1], help="Seeds")
    
    args = parser.parse_args()
    
    # Check GPU availability
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        logger.warning(f"Only {gpu_count} GPU(s) available, using available GPUs")
        gpu_ids = list(range(gpu_count)) if gpu_count > 0 else []
    else:
        gpu_ids = [0, 1]  # Use first two GPUs
    
    if not gpu_ids:
        raise RuntimeError("No CUDA GPUs available")
    
    logger.info(f"Using GPUs: {gpu_ids}")
    
    # Setup
    prompts_file = Path(args.prompts)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    prompts_df = pd.read_csv(prompts_file)
    logger.info(f"Loaded {len(prompts_df)} prompts")
    
    # Prepare generation arguments
    generation_args = {
        'width': args.width,
        'height': args.height,
        'steps': args.steps,
        'guidance_scale': args.guidance_scale,
    }
    
    # Create task and result queues
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Populate task queue
    total_tasks = 0
    for _, row in prompts_df.iterrows():
        prompt_id = row["prompt_id"]
        prompt = row["prompt"]
        for seed in args.seeds:
            task_queue.put((prompt_id, prompt, seed, str(output_dir)))
            total_tasks += 1
    
    # Add shutdown signals
    for _ in gpu_ids:
        task_queue.put(None)
    
    # Start GPU workers
    workers = []
    for gpu_id in gpu_ids:
        worker = threading.Thread(
            target=gpu_worker,
            args=(
                gpu_id, task_queue, result_queue,
                args.model, args.dora_weights, generation_args
            )
        )
        worker.start()
        workers.append(worker)
    
    # Collect results with progress bar
    all_results = []
    completed_tasks = 0
    
    with tqdm(total=total_tasks, desc="Generating images") as pbar:
        while completed_tasks < total_tasks:
            try:
                result = result_queue.get(timeout=30)
                all_results.append(result)
                completed_tasks += 1
                pbar.update(1)
            except queue.Empty:
                logger.warning("No results received in 30 seconds")
                continue
    
    # Wait for all workers to complete
    for worker in workers:
        worker.join()
    
    # Save metadata
    results_df = pd.DataFrame(all_results)
    results_file = output_dir.parent / "generated_dual_gpu.csv"
    results_df.to_csv(results_file, index=False)
    
    logger.info(f"Generated {len(all_results)} images using {len(gpu_ids)} GPUs")
    logger.info(f"Results saved to {results_file}")
    
    # Print performance summary
    gpu_usage = results_df['gpu_id'].value_counts().sort_index()
    logger.info("GPU usage distribution:")
    for gpu_id, count in gpu_usage.items():
        logger.info(f"  GPU {gpu_id}: {count} images")


if __name__ == "__main__":
    main()
