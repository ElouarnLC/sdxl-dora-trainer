#!/usr/bin/env python3
"""
Utility functions for SDXL DoRA training.
"""

import torch
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
from rich.console import Console
from rich.table import Table

console = Console()

def check_gpu_memory():
    """Check available GPU memory."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_total = props.total_memory / (1024**3)  # GB
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        memory_cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB
        memory_free = memory_total - memory_cached
        
        gpu_info.append({
            "device": i,
            "name": props.name,
            "memory_total": memory_total,
            "memory_allocated": memory_allocated,
            "memory_cached": memory_cached,
            "memory_free": memory_free,
            "compute_capability": f"{props.major}.{props.minor}"
        })
    
    return gpu_info

def print_gpu_info():
    """Print GPU information in a nice table."""
    gpu_info = check_gpu_memory()
    
    if "error" in gpu_info:
        console.print(f"[red]GPU Check Failed: {gpu_info['error']}[/red]")
        return
    
    table = Table(title="GPU Information", show_header=True, header_style="bold magenta")
    table.add_column("Device", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Total Memory (GB)", style="yellow")
    table.add_column("Free Memory (GB)", style="blue")
    table.add_column("Compute Capability", style="white")
    
    for gpu in gpu_info:
        table.add_row(
            str(gpu["device"]),
            gpu["name"],
            f"{gpu['memory_total']:.1f}",
            f"{gpu['memory_free']:.1f}",
            gpu["compute_capability"]
        )
    
    console.print(table)

def analyze_dataset(dataset_path: str) -> Dict:
    """Analyze dataset and return statistics."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return {"error": f"Dataset path does not exist: {dataset_path}"}
    
    # Count images and captions
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = []
    captions = []
    
    for file_path in dataset_path.rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            images.append(file_path)
            
            # Check for caption file
            caption_path = file_path.with_suffix('.txt')
            if caption_path.exists():
                captions.append(caption_path)
    
    # Analyze image sizes
    image_sizes = []
    for img_path in images[:100]:  # Sample first 100 images
        try:
            with Image.open(img_path) as img:
                image_sizes.append(img.size)
        except Exception:
            continue
    
    # Calculate statistics
    if image_sizes:
        widths = [size[0] for size in image_sizes]
        heights = [size[1] for size in image_sizes]
        
        stats = {
            "total_images": len(images),
            "total_captions": len(captions),
            "caption_ratio": len(captions) / len(images) if images else 0,
            "image_sizes": {
                "min_width": min(widths),
                "max_width": max(widths),
                "avg_width": sum(widths) / len(widths),
                "min_height": min(heights),
                "max_height": max(heights),
                "avg_height": sum(heights) / len(heights)
            }
        }
    else:
        stats = {
            "total_images": len(images),
            "total_captions": len(captions),
            "caption_ratio": len(captions) / len(images) if images else 0,
            "image_sizes": None
        }
    
    return stats

def print_dataset_info(dataset_path: str):
    """Print dataset information in a nice format."""
    stats = analyze_dataset(dataset_path)
    
    if "error" in stats:
        console.print(f"[red]Dataset Analysis Failed: {stats['error']}[/red]")
        return
    
    console.print(f"\n[bold green]Dataset Analysis for: {dataset_path}[/bold green]")
    console.print(f"Total Images: {stats['total_images']}")
    console.print(f"Total Captions: {stats['total_captions']}")
    console.print(f"Caption Coverage: {stats['caption_ratio']:.1%}")
    
    if stats['image_sizes']:
        console.print("\n[bold blue]Image Size Statistics:[/bold blue]")
        sizes = stats['image_sizes']
        console.print(f"Width: {sizes['min_width']}-{sizes['max_width']} (avg: {sizes['avg_width']:.0f})")
        console.print(f"Height: {sizes['min_height']}-{sizes['max_height']} (avg: {sizes['avg_height']:.0f})")

def estimate_training_time(num_images: int, batch_size: int, max_steps: int, 
                          gradient_accumulation_steps: int = 1) -> Dict:
    """Estimate training time based on dataset size and parameters."""
    
    # Rough estimates based on typical hardware (these can be adjusted)
    time_per_step = {
        "rtx_4090": 3.0,  # seconds per step
        "rtx_3090": 4.5,
        "rtx_3080": 6.0,
        "v100": 5.0,
        "a100": 2.5,
        "h100": 1.5
    }
    
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = num_images // effective_batch_size
    total_epochs = max_steps / steps_per_epoch if steps_per_epoch > 0 else 1
    
    estimates = {}
    for gpu, time_per_step_gpu in time_per_step.items():
        total_time_seconds = max_steps * time_per_step_gpu
        hours = total_time_seconds / 3600
        estimates[gpu] = {
            "total_time_hours": hours,
            "total_time_formatted": f"{int(hours)}h {int((hours % 1) * 60)}m"
        }
    
    return {
        "steps_per_epoch": steps_per_epoch,
        "total_epochs": total_epochs,
        "effective_batch_size": effective_batch_size,
        "gpu_estimates": estimates
    }

def print_training_estimates(num_images: int, batch_size: int, max_steps: int, 
                           gradient_accumulation_steps: int = 1):
    """Print training time estimates."""
    estimates = estimate_training_time(num_images, batch_size, max_steps, gradient_accumulation_steps)
    
    console.print("\n[bold yellow]Training Estimates:[/bold yellow]")
    console.print(f"Steps per epoch: {estimates['steps_per_epoch']}")
    console.print(f"Total epochs: {estimates['total_epochs']:.1f}")
    console.print(f"Effective batch size: {estimates['effective_batch_size']}")
    
    console.print("\n[bold cyan]Estimated Training Time by GPU:[/bold cyan]")
    for gpu, estimate in estimates['gpu_estimates'].items():
        console.print(f"{gpu.upper()}: {estimate['total_time_formatted']}")

def validate_environment():
    """Validate the training environment."""
    issues = []
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        issues.append("Python 3.8 or higher required")
    
    # Check PyTorch
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA not available - GPU training not possible")
        elif torch.version.cuda < "11.0":
            issues.append("CUDA 11.0 or higher recommended")
    except ImportError:
        issues.append("PyTorch not installed")
    
    # Check required libraries
    required_libs = [
        "diffusers", "transformers", "accelerate", "peft", 
        "datasets", "safetensors", "bitsandbytes"
    ]
    
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            issues.append(f"Missing required library: {lib}")
    
    # Check disk space (rough estimate)
    try:
        import shutil
        free_space_gb = shutil.disk_usage(".").free / (1024**3)
        if free_space_gb < 20:
            issues.append(f"Low disk space: {free_space_gb:.1f}GB available (20GB+ recommended)")
    except Exception:
        issues.append("Could not check disk space")
    
    return issues

def print_environment_check():
    """Print environment validation results."""
    console.print("\n[bold magenta]Environment Check[/bold magenta]")
    
    issues = validate_environment()
    
    if not issues:
        console.print("[green]✓ Environment validation passed![/green]")
    else:
        console.print("[red]⚠ Environment issues found:[/red]")
        for issue in issues:
            console.print(f"  • {issue}")
    
    # Print GPU info
    print_gpu_info()

def calculate_optimal_batch_size(gpu_memory_gb: float, resolution: int = 1024) -> int:
    """Calculate optimal batch size based on GPU memory."""
    # Rough estimates for SDXL memory usage
    memory_per_sample = {
        512: 4.0,   # GB per sample at 512x512
        768: 6.0,   # GB per sample at 768x768
        1024: 8.0,  # GB per sample at 1024x1024
    }
    
    memory_needed = memory_per_sample.get(resolution, 8.0)
    
    # Leave some memory for model weights and other operations
    available_memory = gpu_memory_gb * 0.8
    
    optimal_batch_size = max(1, int(available_memory // memory_needed))
    
    return optimal_batch_size

def suggest_training_params(dataset_path: str, gpu_memory_gb: Optional[float] = None):
    """Suggest optimal training parameters based on dataset and hardware."""
    
    # Analyze dataset
    dataset_stats = analyze_dataset(dataset_path)
    if "error" in dataset_stats:
        console.print(f"[red]Cannot analyze dataset: {dataset_stats['error']}[/red]")
        return
    
    num_images = dataset_stats['total_images']
    
    # Get GPU memory if not provided
    if gpu_memory_gb is None:
        gpu_info = check_gpu_memory()
        if isinstance(gpu_info, list) and len(gpu_info) > 0:
            gpu_memory_gb = gpu_info[0]['memory_total']
        else:
            gpu_memory_gb = 12.0  # Default assumption
    
    # Calculate suggestions
    optimal_batch_size = calculate_optimal_batch_size(gpu_memory_gb)
    
    # Suggest training steps based on dataset size
    if num_images < 100:
        suggested_steps = 1500
    elif num_images < 500:
        suggested_steps = 1000
    else:
        suggested_steps = max(500, num_images // 2)
    
    # Suggest rank based on dataset complexity
    if num_images < 50:
        suggested_rank = 32
    elif num_images < 200:
        suggested_rank = 64
    else:
        suggested_rank = 128
    
    console.print("\n[bold green]Suggested Training Parameters:[/bold green]")
    console.print(f"Batch Size: {optimal_batch_size}")
    console.print(f"Max Training Steps: {suggested_steps}")
    console.print(f"DoRA Rank: {suggested_rank}")
    console.print(f"DoRA Alpha: {suggested_rank // 2}")
    
    if optimal_batch_size == 1:
        console.print("Gradient Accumulation Steps: 4-8 (to increase effective batch size)")
    
    # Print training time estimates
    print_training_estimates(num_images, optimal_batch_size, suggested_steps)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SDXL DoRA Training Utilities")
    parser.add_argument("command", choices=["check-env", "analyze-dataset", "suggest-params"],
                       help="Command to run")
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--gpu-memory", type=float, help="GPU memory in GB")
    
    args = parser.parse_args()
    
    if args.command == "check-env":
        print_environment_check()
    elif args.command == "analyze-dataset":
        if not args.dataset:
            console.print("[red]--dataset required for analyze-dataset command[/red]")
        else:
            print_dataset_info(args.dataset)
    elif args.command == "suggest-params":
        if not args.dataset:
            console.print("[red]--dataset required for suggest-params command[/red]")
        else:
            suggest_training_params(args.dataset, args.gpu_memory)
