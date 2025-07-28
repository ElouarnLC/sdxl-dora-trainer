#!/usr/bin/env python3
"""
Simple DoRA DSPO Training Script

This script can be run from the diffusion_rlhf directory and will work with your existing prompts.csv file.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the current directory to Python path (we're already in diffusion_rlhf)
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Try to import DSPOFineTuner with better error handling
try:
    from dspo.tuner import DSPOFineTuner
    DSPO_AVAILABLE = True
except ImportError as e:
    DSPO_AVAILABLE = False
    DSPO_IMPORT_ERROR = str(e)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SimplePreferencePairDataset(Dataset):
    """Simple dataset for DSPO preference pairs."""
    
    def __init__(self, prompts: list[str], num_pairs: int = 100):
        """Create synthetic preference pairs from prompts."""
        self.pairs = []
        
        # Create simple preference pairs
        # In a real implementation, you'd generate actual images and score them
        for i in range(min(num_pairs, len(prompts))):
            prompt = prompts[i % len(prompts)]
            
            # Create dummy preference pair
            self.pairs.append({
                'prompt': prompt,
                'img_pos': torch.randn(3, 1024, 1024),  # Dummy positive image
                'img_neg': torch.randn(3, 1024, 1024),  # Dummy negative image
            })
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]


def main():
    """Simple DoRA DSPO training."""
    parser = argparse.ArgumentParser(description="Simple DoRA DSPO Training")
    parser.add_argument("--prompts", type=str, default="data/prompts.csv",
                       help="Path to prompts CSV file")
    parser.add_argument("--reward-model", type=str, default="outputs/full_enhanced_final/best/model.pt",
                       help="Path to trained reward model")
    parser.add_argument("--output", type=str, default="outputs/dspo_simple",
                       help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=50,
                       help="Number of preference pairs")
    parser.add_argument("--num-steps", type=int, default=1000,
                       help="Number of training steps")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--rank", type=int, default=16,
                       help="DoRA rank")
    
    args = parser.parse_args()
    
    logger.info("🚀 Simple DoRA DSPO Training")
    logger.info("=" * 50)
    
    # Check if prompts file exists
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        logger.error(f"❌ Prompts file not found: {prompts_path}")
        logger.info("Available files in data/:")
        data_dir = Path("data")
        if data_dir.exists():
            for file in data_dir.iterdir():
                if file.is_file():
                    logger.info(f"  - {file}")
        else:
            logger.info("  No data/ directory found")
        
        logger.info("\nTrying alternative locations...")
        alt_paths = [
            "data/prompts_example.csv",
            "data/prompts_cordeliers.csv"
        ]
        
        for alt_path in alt_paths:
            if Path(alt_path).exists():
                logger.info(f"✅ Found: {alt_path}")
                args.prompts = alt_path
                break
        else:
            logger.error("❌ No prompts file found!")
            return
    
    # Check if reward model exists
    reward_model_path = Path(args.reward_model)
    if not reward_model_path.exists():
        logger.error(f"❌ Reward model not found: {reward_model_path}")
        logger.info("\n💡 Train your reward model first:")
        logger.info("  python scripts/train_small_dataset.py \\")
        logger.info("    --data data/annotations/multihead_ratings.csv \\")
        logger.info("    --prompts data/prompts.csv \\")
        logger.info("    --output outputs/full_enhanced_final")
        return
    
    # Load prompts
    logger.info(f"📝 Loading prompts from: {args.prompts}")
    try:
        if args.prompts.endswith('.csv'):
            df = pd.read_csv(args.prompts)
            if 'prompt' in df.columns:
                prompts = df['prompt'].tolist()
            else:
                prompts = df.iloc[:, 0].tolist()
        else:
            with open(args.prompts) as f:
                prompts = [line.strip() for line in f if line.strip()]
        
        # Remove empty prompts
        prompts = [p for p in prompts if p.strip()]
        logger.info(f"✅ Loaded {len(prompts)} prompts")
        
    except Exception as e:
        logger.error(f"❌ Error loading prompts: {e}")
        return
    
    # Create dataset
    dataset = SimplePreferencePairDataset(prompts, args.num_pairs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    logger.info(f"📊 Created {len(dataset)} preference pairs")
    
    # Initialize DSPO fine-tuner
    logger.info("🔧 Initializing DSPOFineTuner...")
    
    if not DSPO_AVAILABLE:
        logger.error(f"❌ DSPO not available: {DSPO_IMPORT_ERROR}")
        logger.info("\n💡 To fix this, make sure you have installed the required dependencies:")
        logger.info("   pip install open-clip-torch>=2.24.0 peft>=0.7.0 accelerate>=0.24.0")
        logger.info("   pip install diffusers>=0.24.0 transformers>=4.35.0")
        logger.info("   pip install safetensors>=0.4.0")
        logger.info("\nAlso check that you're in the correct environment:")
        logger.info(f"   Current directory: {current_dir}")
        return
    
    try:
        tuner = DSPOFineTuner(
            model_name="stabilityai/stable-diffusion-xl-base-1.0",
            rank=args.rank,
            alpha=args.rank * 4,  # Standard ratio
            learning_rate=args.lr,
            beta=0.1,
            mixed_precision="fp16" if torch.cuda.is_available() else "no",
            gradient_checkpointing=True,
        )
        
        logger.info("✅ DSPOFineTuner initialized")
        logger.info(f"🎯 Using reward model: {args.reward_model}")
        
        # Note: In a real implementation, you would load the reward model and perform training
        logger.info("\n⚠️  Note: This is a demo version")
        logger.info("To perform actual DSPO training, you would:")
        logger.info(f"1. Load reward model from: {args.reward_model}")
        logger.info("2. Generate real preference pairs using the reward model")
        logger.info("3. Run the DSPO training loop")
        
        logger.info(f"\n🎯 Training would proceed with {args.num_steps} steps")
        logger.info(f"📂 Output directory: {args.output}")
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("\n💡 To fix this:")
        logger.info("1. Install required packages:")
        logger.info("   pip install peft>=0.8.0 accelerate diffusers transformers")
        logger.info("2. Make sure you're in the correct environment")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
