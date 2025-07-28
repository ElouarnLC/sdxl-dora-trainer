#!/usr/bin/env python3
"""
Quick launcher for DoRA DSPO training.

This script provides an easy way to start DoRA-based DSPO training
with your high-performance reward model.
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if all required components are available."""
    checks = []
    
    # Check reward model
    reward_model_path = Path("outputs/full_enhanced_final/best/model.pt")
    if reward_model_path.exists():
        checks.append("✅ Reward model found")
    else:
        checks.append("❌ Reward model not found - run enhanced training first")
        return False
    
    # Check prompts
    prompts_path = Path("diffusion_rlhf/data/prompts.csv")
    if prompts_path.exists():
        checks.append("✅ Example prompts found")
    else:
        checks.append("⚠️ Example prompts not found - will use provided path")
    
    # Check DSPO trainer
    dspo_script = Path("diffusion_rlhf/scripts/train_dspo_with_dora.py")
    if dspo_script.exists():
        checks.append("✅ DoRA DSPO trainer ready")
    else:
        checks.append("❌ DoRA DSPO trainer script missing")
        return False
    
    for check in checks:
        logger.info(check)
    
    return True


def create_quick_config():
    """Create a quick configuration for DoRA DSPO training."""
    return {
        "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        "reward_model": "outputs/full_enhanced_final/best/model.pt",
        "prompts": "diffusion_rlhf/data/prompts.csv",
        "output": "outputs/dspo_dora_quick",
        "num_pairs": 50,  # Start small for testing
        "epochs": 5,
        "lr": 1e-5,
        "dora_rank": 16,  # Conservative for initial run
        "dora_alpha": 16.0,
    }


def main():
    """Quick launcher for DoRA DSPO training."""
    logger.info("🚀 DoRA DSPO Training Quick Launcher")
    logger.info("=" * 50)
    
    # Check requirements
    if not check_requirements():
        logger.error("❌ Requirements check failed!")
        logger.info("\n📋 To fix:")
        logger.info("1. Run enhanced reward model training first")
        logger.info("2. Ensure all required files are present")
        sys.exit(1)
    
    # Get configuration
    config = create_quick_config()
    
    logger.info("\n⚙️ Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Build command
    cmd_args = [
        "python", "diffusion_rlhf/scripts/train_dspo_with_dora.py",
        "--prompts", str(config["prompts"]),
        "--output", str(config["output"]),
        "--reward-model", str(config["reward_model"]),
        "--model-name", str(config["model_name"]),
        "--num-pairs", str(config["num_pairs"]),
        "--epochs", str(config["epochs"]),
        "--lr", str(config["lr"]),
    ]
    
    logger.info("\n🎯 Ready to start DoRA DSPO training!")
    logger.info("Command to run:")
    logger.info(" ".join(cmd_args))
    
    print("\n" + "="*60)
    print("🎮 MANUAL EXECUTION REQUIRED")
    print("="*60)
    print("\nTo start training, run this command:")
    print(f"  {' '.join(cmd_args)}")
    print("\nOr with custom prompts:")
    print(f"  python diffusion_rlhf/scripts/train_dspo_with_dora.py \\")
    print(f"    --prompts YOUR_PROMPTS.csv \\")
    print(f"    --output outputs/dspo_dora_custom \\")
    print(f"    --num-pairs 100 \\")
    print(f"    --epochs 10")
    
    print("\n📊 Expected results:")
    print("  • DoRA fine-tuned SDXL model")
    print("  • Preference optimization with F1=0.8333 reward model")
    print("  • Improved image quality and prompt alignment")
    
    print("\n💾 Outputs will be saved to:")
    print(f"  {config['output']}/")
    print("    ├── dora_weights/")
    print("    └── training_state.pt")


if __name__ == "__main__":
    main()
