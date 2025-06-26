#!/usr/bin/env python3
"""
Example usage of SDXL DoRA Trainer.
This script demonstrates how to use the trainer with different configurations.
"""

import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from sdxl_dora_trainer import TrainingConfig
from utils import print_environment_check

def example_basic_training():
    """Example of basic training setup."""
    print("=== Basic Training Example ===")
    
    # Create a basic configuration
    config = TrainingConfig()
    config.dataset_path = "./datasets/my_images"  # Change this to your dataset
    config.output_dir = "./output/basic_training"
    config.max_train_steps = 500
    config.rank = 32
    config.alpha = 16
    config.learning_rate = 1e-4
    
    print(f"Dataset: {config.dataset_path}")
    print(f"Output: {config.output_dir}")
    print(f"Steps: {config.max_train_steps}")
    print(f"DoRA Rank: {config.rank}")
    print(f"Learning Rate: {config.learning_rate}")
    
    # Note: Don't actually run training in this example
    # trainer = DoRATrainer(config)
    # trainer.train()
    
    print("To run this training, uncomment the trainer lines above")

def example_advanced_training():
    """Example of advanced training setup."""
    print("\n=== Advanced Training Example ===")
    
    config = TrainingConfig()
    
    # Dataset and output
    config.dataset_path = "./datasets/portraits"
    config.output_dir = "./output/advanced_training"
    
    # DoRA parameters for better quality
    config.rank = 128
    config.alpha = 64
    config.dropout = 0.1
    
    # Training parameters
    config.learning_rate = 5e-5
    config.batch_size = 2
    config.gradient_accumulation_steps = 2
    config.max_train_steps = 2000
    config.warmup_steps = 200
    
    # Image parameters
    config.resolution = 1024
    config.center_crop = True
    config.random_flip = True
    
    # Memory optimization
    config.mixed_precision = "fp16"
    config.gradient_checkpointing = True
    config.use_8bit_adam = True
    
    # Logging
    config.report_to = "wandb"
    config.project_name = "portrait-dora-training"
    config.run_name = "advanced-run-1"
    
    # Validation
    config.validation_prompts = [
        "a professional portrait of a person",
        "headshot photography, studio lighting",
        "portrait in natural lighting"
    ]
    config.validation_steps = 100
    
    print(f"Dataset: {config.dataset_path}")
    print(f"Output: {config.output_dir}")
    print(f"DoRA Rank: {config.rank}, Alpha: {config.alpha}")
    print(f"Batch Size: {config.batch_size} (effective: {config.batch_size * config.gradient_accumulation_steps})")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Resolution: {config.resolution}x{config.resolution}")
    print(f"Mixed Precision: {config.mixed_precision}")
    print(f"Logging: {config.report_to}")
    
    print("To run this training, create a DoRATrainer with this config")

def example_style_training():
    """Example of style-specific training."""
    print("\n=== Style Training Example ===")
    
    config = TrainingConfig()
    
    # Dataset for a specific art style
    config.dataset_path = "./datasets/anime_style"
    config.output_dir = "./output/anime_style_training"
    
    # Higher rank for style learning
    config.rank = 256
    config.alpha = 128
    config.dropout = 0.05
    
    # Conservative learning rate for style preservation
    config.learning_rate = 3e-5
    config.batch_size = 1
    config.gradient_accumulation_steps = 8
    config.max_train_steps = 3000
    
    # Style-specific validation prompts
    config.validation_prompts = [
        "anime style portrait",
        "manga character illustration",
        "japanese animation art style"
    ]
    
    print(f"Dataset: {config.dataset_path}")
    print("Style: Anime/Manga")
    print(f"DoRA Rank: {config.rank} (high for style learning)")
    print(f"Learning Rate: {config.learning_rate} (conservative)")
    print(f"Training Steps: {config.max_train_steps}")

def example_batch_configuration():
    """Example of batch training configuration."""
    print("\n=== Batch Training Example ===")
    
    batch_config = {
        "base_config": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "output_dir": "./batch_output",
            "max_train_steps": 1000,
            "resolution": 1024,
            "mixed_precision": "fp16",
            "gradient_checkpointing": True,
            "report_to": "tensorboard"
        },
        "jobs": [
            {
                "name": "portraits_rank32",
                "dataset_path": "./datasets/portraits",
                "rank": 32,
                "alpha": 16,
                "learning_rate": 1e-4
            },
            {
                "name": "portraits_rank64",
                "dataset_path": "./datasets/portraits", 
                "rank": 64,
                "alpha": 32,
                "learning_rate": 1e-4
            },
            {
                "name": "landscapes_rank64",
                "dataset_path": "./datasets/landscapes",
                "rank": 64,
                "alpha": 32,
                "learning_rate": 5e-5
            }
        ]
    }
    
    print("Batch configuration for comparing different ranks:")
    for job in batch_config["jobs"]:
        print(f"  - {job['name']}: rank={job['rank']}, lr={job['learning_rate']}")
    
    print("\nSave this as batch_config.yaml and run:")
    print("python batch_train.py --config batch_config.yaml")

def example_inference():
    """Example of inference usage."""
    print("\n=== Inference Example ===")
    
    print("After training, you can generate images using:")
    print("python inference.py --dora_weights ./output/checkpoints/checkpoint-1000 --prompt 'your prompt here'")
    print()
    print("Interactive mode:")
    print("python inference.py --dora_weights ./output/checkpoints/checkpoint-1000 --interactive")
    print()
    print("Multiple prompts:")
    print("python inference.py --dora_weights ./output/checkpoints/checkpoint-1000 \\")
    print("  --prompt 'first prompt' --prompt 'second prompt' --prompt 'third prompt'")
    print()
    print("From file:")
    print("python inference.py --dora_weights ./output/checkpoints/checkpoint-1000 \\")
    print("  --prompts_file prompts.txt --output_dir ./my_generated_images")

def main():
    """Run all examples."""
    print("SDXL DoRA Trainer - Usage Examples")
    print("=" * 50)
    
    # Check environment first
    print("Environment Check:")
    print_environment_check()
    
    # Show examples
    example_basic_training()
    example_advanced_training()
    example_style_training()
    example_batch_configuration()
    example_inference()
    
    print("\n" + "=" * 50)
    print("Getting Started:")
    print("1. Prepare your dataset in a folder with images and caption files")
    print("2. Check your environment: python utils.py check-env")
    print("3. Analyze your dataset: python utils.py analyze-dataset --dataset /path/to/images")
    print("4. Get suggestions: python utils.py suggest-params --dataset /path/to/images")
    print("5. Start training: python sdxl_dora_trainer.py --dataset_path /path/to/images")
    print("6. Generate images: python inference.py --dora_weights /path/to/checkpoint --interactive")

if __name__ == "__main__":
    main()
