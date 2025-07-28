#!/usr/bin/env python3
"""
Quick start script for improved reward model training.

This script provides an easy way to test the enhanced model with better defaults.
"""

import argparse
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_enhanced_training(
    data_file: str,
    prompts_file: str,
    output_dir: str = "outputs/enhanced_reward_model",
    quick_test: bool = False
):
    """
    Run enhanced training with optimized settings.
    
    Parameters
    ----------
    data_file : str
        Path to your data file (ratings.csv or pairs.jsonl)
    prompts_file : str
        Path to prompts.csv
    output_dir : str
        Output directory
    quick_test : bool
        If True, uses reduced epochs for quick testing
    """
    
    # Determine the best configuration based on your needs
    if quick_test:
        config = {
            "model_name": "ViT-L-14",  # Faster for testing
            "fusion_method": "attention",
            "loss_type": "focal",
            "hidden_dims": [512, 256],
            "dropout": 0.15,
            "learning_rate": 5e-5,
            "batch_size": 8,
            "num_epochs": 5,  # Quick test
            "min_rating_diff": 1.0,
            "mixup_alpha": 0.2,
            "ema_decay": 0.999,
        }
    else:
        config = {
            "model_name": "ViT-L-14-336",  # Higher resolution for better performance
            "fusion_method": "attention",
            "loss_type": "focal",
            "hidden_dims": [768, 384, 192],
            "dropout": 0.15,
            "learning_rate": 5e-5,
            "weight_decay": 0.02,
            "batch_size": 16,
            "num_epochs": 20,
            "gradient_accumulation_steps": 2,
            "max_grad_norm": 1.0,
            "warmup_steps": 500,
            "min_rating_diff": 1.0,
            "mixup_alpha": 0.2,
            "ema_decay": 0.999,
            "patience": 5,
            "save_every": 5,
        }
    
    logger.info("Starting enhanced reward model training...")
    logger.info(f"Data file: {data_file}")
    logger.info(f"Prompts file: {prompts_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Import and run the enhanced training
    try:
        from train_enhanced_multimodal_reward import train_enhanced_multimodal_reward_model
        
        train_enhanced_multimodal_reward_model(
            data_file=data_file,
            prompts_file=prompts_file,
            output_dir=output_dir,
            config=config
        )
        
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"📁 Model saved to: {output_dir}")
        logger.info(f"🏆 Best model saved to: {output_dir}/best")
        
    except ImportError:
        logger.error("❌ Enhanced training script not found. Please ensure you have:")
        logger.error("  1. enhanced_multimodal_reward.py in dspo/ directory")
        logger.error("  2. train_enhanced_multimodal_reward.py in scripts/ directory")
        
        # Fallback to original training
        logger.info("🔄 Falling back to original training with improved settings...")
        fallback_command = f"""
python scripts/train_multimodal_reward.py \\
    --mode multihead \\
    --ratings {data_file} \\
    --prompts {prompts_file} \\
    --output {output_dir}_fallback \\
    --epochs {config['num_epochs']} \\
    --batch-size {config['batch_size']} \\
    --learning-rate {config['learning_rate']} \\
    --fusion-method cross_attention \\
    --min-rating-diff {config['min_rating_diff']} \\
    --mixed-precision bf16
"""
        logger.info(f"Run this command:\n{fallback_command}")


def analyze_current_performance(model_dir: str):
    """
    Analyze current model performance and suggest improvements.
    
    Parameters
    ----------
    model_dir : str
        Directory containing trained model
    """
    
    model_path = Path(model_dir)
    if not model_path.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return
    
    logger.info(f"📊 Analyzing model performance in: {model_dir}")
    
    # Check for training logs
    log_files = list(model_path.glob("**/logs/**/*.log"))
    if log_files:
        logger.info(f"Found {len(log_files)} log files")
        # You could parse logs here to extract metrics
    
    # Check for checkpoints
    checkpoints = list(model_path.glob("checkpoint-*"))
    if checkpoints:
        logger.info(f"Found {len(checkpoints)} checkpoints")
    
    # Suggestions based on current F1 score
    suggestions = [
        "🎯 Quick improvements to try:",
        "  1. Switch to enhanced model with attention fusion",
        "  2. Use focal loss instead of BCE loss",
        "  3. Increase min_rating_diff to 1.0 for clearer preferences",
        "  4. Add data augmentation with MixUp",
        "  5. Use higher resolution backbone (ViT-L-14-336)",
        "",
        "🚀 For F1 > 0.8, implement:",
        "  1. Multi-scale feature extraction",
        "  2. Advanced learning rate scheduling",
        "  3. Exponential moving average (EMA)",
        "  4. Better data quality and balanced sampling",
        "",
        "📈 Monitoring tips:",
        "  1. Track multiple metrics (F1, AUC, confidence)",
        "  2. Use validation metrics for model selection",
        "  3. Analyze prediction errors for insights",
        "  4. Visualize learned head importance weights",
    ]
    
    for suggestion in suggestions:
        logger.info(suggestion)


def main():
    parser = argparse.ArgumentParser(description="Enhanced reward model training launcher")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train enhanced reward model")
    train_parser.add_argument("--data", required=True, help="Path to data file")
    train_parser.add_argument("--prompts", required=True, help="Path to prompts.csv")
    train_parser.add_argument("--output", default="outputs/enhanced_reward", help="Output directory")
    train_parser.add_argument("--quick-test", action="store_true", help="Quick test with reduced epochs")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze model performance")
    analyze_parser.add_argument("--model-dir", required=True, help="Model directory to analyze")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show recommended configurations")
    
    args = parser.parse_args()
    
    if args.command == "train":
        run_enhanced_training(
            data_file=args.data,
            prompts_file=args.prompts,
            output_dir=args.output,
            quick_test=args.quick_test
        )
    
    elif args.command == "analyze":
        analyze_current_performance(args.model_dir)
    
    elif args.command == "config":
        logger.info("📋 Recommended configurations:")
        
        configs = {
            "Quick Test (5 epochs, fast)": {
                "model_name": "ViT-L-14",
                "fusion_method": "attention",
                "loss_type": "focal",
                "batch_size": 8,
                "num_epochs": 5,
                "min_rating_diff": 1.0,
            },
            
            "Balanced (good performance/speed)": {
                "model_name": "ViT-L-14-336",
                "fusion_method": "attention", 
                "loss_type": "focal",
                "batch_size": 16,
                "num_epochs": 15,
                "min_rating_diff": 1.0,
                "mixup_alpha": 0.2,
            },
            
            "High Performance (best F1 score)": {
                "model_name": "ViT-L-14-336",
                "fusion_method": "attention",
                "loss_type": "focal",
                "hidden_dims": [768, 384, 192],
                "batch_size": 16,
                "num_epochs": 25,
                "min_rating_diff": 1.5,
                "mixup_alpha": 0.2,
                "ema_decay": 0.999,
            }
        }
        
        for name, config in configs.items():
            logger.info(f"\n{name}:")
            for key, value in config.items():
                logger.info(f"  {key}: {value}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
