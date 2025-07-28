#!/usr/bin/env python3
"""
DSPO training script using the trained enhanced reward model.

This script trains an SDXL LoRA model using Direct Statistical Preference Optimization (DSPO)
with your high-performance reward model (F1=0.8333).
"""

import argparse
import logging
import os
import torch
from pathlib import Path
from typing import Dict, Any

from accelerate import Accelerator
from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model
import wandb

from dspo.trained_reward_model import load_reward_model
from dspo.datasets import PreferenceDataset
from dspo.tuner import DSPOTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_dspo_config() -> Dict[str, Any]:
    """
    Create DSPO training configuration optimized for your reward model.
    
    Returns
    -------
    Dict[str, Any]
        DSPO configuration
    """
    return {
        # Model configuration
        "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        "revision": "main",
        "variant": "fp16",
        
        # LoRA configuration
        "lora_rank": 64,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": [
            "to_k", "to_q", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
            "conv1", "conv2", "conv_shortcut",
        ],
        
        # Training configuration
        "num_train_epochs": 10,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "cosine_with_restarts",
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        
        # DSPO-specific parameters
        "beta": 0.1,  # DSPO temperature parameter
        "label_smoothing": 0.0,
        "loss_type": "sigmoid",  # or "hinge"
        
        # Generation parameters
        "resolution": 1024,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        
        # Reward model integration
        "reward_model_path": "outputs/full_enhanced_final/best/model.pt",
        "reward_scale": 1.0,
        
        # Logging and saving
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 200,
        "output_dir": "outputs/dspo_sdxl_lora",
        "run_name": "dspo_enhanced_reward_f1_0833",
        
        # Data configuration
        "max_train_samples": None,
        "dataloader_num_workers": 4,
        "seed": 42,
    }


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging and wandb if available."""
    if "WANDB_API_KEY" in os.environ:
        wandb.init(
            project="sdxl-dspo-enhanced-reward",
            name=config["run_name"],
            config=config,
            tags=["dspo", "sdxl", "enhanced-reward", "f1-0833"]
        )
        logger.info("✅ Weights & Biases logging enabled")
    else:
        logger.info("ℹ️ W&B logging disabled (no API key found)")


def create_lora_model(base_model, config: Dict[str, Any]):
    """
    Create LoRA model from base SDXL model.
    
    Parameters
    ----------
    base_model : StableDiffusionXLPipeline
        Base SDXL model
    config : Dict[str, Any]
        Configuration dictionary
        
    Returns
    -------
    LoRA model
    """
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        task_type="DIFFUSION",
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(base_model.unet, lora_config)
    logger.info(f"✅ LoRA applied to UNet with rank={config['lora_rank']}")
    
    return unet


def train_dspo_with_enhanced_reward(
    config: Dict[str, Any],
    data_file: str,
    prompts_file: str,
) -> None:
    """
    Train SDXL LoRA model using DSPO with enhanced reward model.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Training configuration
    data_file : str
        Path to preference data
    prompts_file : str
        Path to prompts file
    """
    # Setup
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision="fp16",
        log_with="wandb" if "WANDB_API_KEY" in os.environ else None,
        project_dir=config["output_dir"],
    )
    
    device = accelerator.device
    logger.info(f"🚀 Starting DSPO training on {device}")
    
    # Load enhanced reward model
    logger.info("📊 Loading enhanced reward model...")
    reward_model = load_reward_model(config["reward_model_path"])
    logger.info(f"✅ Reward model loaded with F1={reward_model.metrics.get('f1', 0):.4f}")
    
    # Load base SDXL model
    logger.info("🎨 Loading SDXL base model...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        config["model_name"],
        revision=config["revision"],
        variant=config["variant"],
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    
    # Create LoRA model
    logger.info("🔧 Setting up LoRA...")
    lora_unet = create_lora_model(pipeline, config)
    
    # Load dataset
    logger.info("📁 Loading preference dataset...")
    dataset = PreferenceDataset(
        data_file=data_file,
        prompts_file=prompts_file,
        resolution=config["resolution"],
    )
    logger.info(f"📊 Dataset loaded: {len(dataset)} preference pairs")
    
    # Create DSPO trainer
    logger.info("🏋️ Setting up DSPO trainer...")
    trainer = DSPOTrainer(
        model=lora_unet,
        pipeline=pipeline,
        reward_model=reward_model,
        config=config,
        accelerator=accelerator,
    )
    
    # Train
    logger.info("🚀 Starting DSPO training...")
    trainer.train(dataset)
    
    # Save final model
    output_path = Path(config["output_dir"])
    final_path = output_path / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    
    lora_unet.save_pretrained(final_path)
    pipeline.save_pretrained(final_path / "pipeline")
    
    logger.info(f"✅ Training completed! Model saved to {final_path}")
    logger.info("🎨 Your SDXL model is now optimized with your F1=0.8333 reward model!")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="DSPO training with enhanced reward model")
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to preference data file"
    )
    parser.add_argument(
        "--prompts", 
        type=str,
        required=True,
        help="Path to prompts file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dspo_enhanced_sdxl",
        help="Output directory"
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default="outputs/full_enhanced_final/best/model.pt",
        help="Path to trained reward model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_dspo_config()
    config.update({
        "output_dir": args.output_dir,
        "reward_model_path": args.reward_model,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    })
    
    # Validate inputs
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    if not Path(args.prompts).exists():
        raise FileNotFoundError(f"Prompts file not found: {args.prompts}")
    if not Path(args.reward_model).exists():
        raise FileNotFoundError(f"Reward model not found: {args.reward_model}")
    
    # Setup logging
    setup_logging(config)
    
    # Log configuration
    logger.info("🔧 DSPO Training Configuration:")
    logger.info(f"   📊 Data: {args.data}")
    logger.info(f"   🏆 Reward Model: {args.reward_model}")
    logger.info(f"   🎨 Base Model: {config['model_name']}")
    logger.info(f"   🔧 LoRA Rank: {config['lora_rank']}")
    logger.info(f"   📈 Learning Rate: {config['learning_rate']}")
    logger.info(f"   🔄 Epochs: {config['num_train_epochs']}")
    logger.info(f"   💾 Output: {args.output_dir}")
    
    # Start training
    try:
        train_dspo_with_enhanced_reward(config, args.data, args.prompts)
        logger.info("🎉 DSPO training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
