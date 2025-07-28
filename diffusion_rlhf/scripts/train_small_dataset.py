#!/usr/bin/env python3
"""
Optimized training script for small datasets.

This script is specifically designed for training multimodal reward models 
on small datasets with improved hyperparameters and reduced overfitting.
"""

import argparse
import logging
from pathlib import Path

from train_enhanced_multimodal_reward import train_enhanced_multimodal_reward_model
from dspo.enhanced_multimodal_reward import create_enhanced_training_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_small_dataset_config() -> dict:
    """Create configuration optimized for small datasets."""
    config = create_enhanced_training_config()
    
    # Override with small dataset specific settings
    config.update({
        # More aggressive learning
        "learning_rate": 2e-4,         # Higher learning rate
        "freeze_backbone": False,      # Unfreeze backbone for better learning
        
        # Better for small datasets
        "batch_size": 4,               # Smaller batches
        "gradient_accumulation_steps": 8,  # Higher accumulation
        "num_epochs": 50,              # More epochs
        "warmup_steps": 50,            # Minimal warmup
        
        # Reduced regularization
        "dropout": 0.05,               # Very low dropout
        "mixup_alpha": 0.05,           # Minimal mixup
        "weight_decay": 0.005,         # Low weight decay
        
        # Better validation
        "val_split": 0.1,              # Minimal validation split
        "patience": 10,                # More patience
        "min_rating_diff": 0.3,        # Lower threshold for more pairs
        
        # EMA
        "ema_decay": 0.95,             # Fast EMA for small datasets
    })
    
    return config


def main() -> None:
    """Main function optimized for small datasets."""
    parser = argparse.ArgumentParser(
        description="Train multimodal reward model on small datasets"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data file (ratings.csv or pairs.jsonl)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to prompts.csv file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/small_dataset_model",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    
    args = parser.parse_args()
    
    # Get optimized configuration
    config = create_small_dataset_config()
    
    # Override with command line arguments
    config.update({
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
    })
    
    # Validate inputs
    data_file = Path(args.data)
    prompts_file = Path(args.prompts)
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    logger.info("🚀 Starting optimized training for small datasets")
    logger.info(f"📊 Data: {data_file}")
    logger.info(f"📝 Prompts: {prompts_file}")
    logger.info(f"💾 Output: {args.output}")
    logger.info(f"⚙️ Config highlights:")
    logger.info(f"   • Learning rate: {config['learning_rate']}")
    logger.info(f"   • Backbone frozen: {config['freeze_backbone']}")
    logger.info(f"   • Batch size: {config['batch_size']}")
    logger.info(f"   • Gradient accumulation: {config['gradient_accumulation_steps']}")
    logger.info(f"   • Epochs: {config['num_epochs']}")
    
    # Start training
    try:
        train_enhanced_multimodal_reward_model(
            data_file=str(data_file),
            prompts_file=str(prompts_file),
            output_dir=args.output,
            config=config,
        )
        logger.info("✅ Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
