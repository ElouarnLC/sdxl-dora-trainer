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
    
    # Check reward model (flexible paths)
    reward_model_paths = [
        Path("outputs/full_enhanced_final/best/model.pt"),
        Path("diffusion_rlhf/outputs/full_enhanced_final/best/model.pt"),
    ]
    reward_model_found = False
    for path in reward_model_paths:
        if path.exists():
            checks.append("✅ Reward model found")
            reward_model_found = True
            break
    
    if not reward_model_found:
        checks.append("❌ Reward model not found - run enhanced training first")
        return False
    
    # Check prompts (flexible paths)
    prompts_paths = [
        Path("diffusion_rlhf/data/prompts_example.csv"),
        Path("diffusion_rlhf/data/prompts_cordeliers.csv"),
        Path("data/prompts.csv"),  # User mentioned this path
    ]
    prompts_found = False
    for path in prompts_paths:
        if path.exists():
            checks.append(f"✅ Example prompts found: {path}")
            prompts_found = True
            break
    
    if not prompts_found:
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
        "prompts": "diffusion_rlhf/data/prompts_example.csv",  # Use existing prompts
        "output": "outputs/dspo_dora_quick",
        "num_pairs": 50,  # Start small for testing
        "num_steps": 1000,  # Use steps instead of epochs
        "lr": 5e-5,
        "rank": 16,  # Conservative for initial run
        "alpha": 64,  # 4 * rank as default
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
        "--num-steps", str(config["num_steps"]),  # Use steps instead of epochs
        "--rank", str(config["rank"]),
        "--alpha", str(config["alpha"]),
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
