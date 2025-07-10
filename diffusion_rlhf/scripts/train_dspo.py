#!/usr/bin/env python3
"""
Train DSPO model for preference optimization of SDXL.

Fine-tunes SDXL model using DoRA adapters and preference optimization
with a pre-trained reward model.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dspo.datasets import PairPreferenceDataset
from dspo.tuner import DSPOFineTuner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_dataloader(
    pairs_file: str,
    batch_size: int = 8,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create training dataloader for DSPO.
    
    Parameters
    ----------
    pairs_file : str
        Path to pairs.jsonl file
    batch_size : int
        Batch size
    num_workers : int
        Number of dataloader workers
        
    Returns
    -------
    DataLoader
        Training dataloader
    """
    dataset = PairPreferenceDataset(
        pairs_file=pairs_file,
        split="train",
        image_size=1024,  # SDXL resolution
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    logger.info(f"Created dataloader with {len(dataset)} samples")
    return dataloader


def train_dspo_model(
    reward_model_path: str,
    pairs_file: str,
    output_dir: str,
    rank: int = 8,
    alpha: int | None = None,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    batch_size: int = 8,
    num_steps: int = 2000,
    log_every: int = 100,
    save_every: int = 500,
    sample_every: int = 500,
    mixed_precision: str = "no",
    gradient_checkpointing: bool = True,
) -> None:
    """
    Train DSPO model.
    
    Parameters
    ----------
    reward_model_path : str
        Path to trained reward model
    pairs_file : str
        Path to pairs.jsonl file
    output_dir : str
        Output directory for model checkpoints
    rank : int, default=8
        DoRA rank parameter
    alpha : int | None, default=None
        DoRA alpha parameter (defaults to 4 * rank)
    learning_rate : float, default=5e-5
        Learning rate
    beta : float, default=0.1
        Preference optimization beta parameter
    batch_size : int, default=8
        Batch size
    num_steps : int, default=2000
        Number of training steps
    log_every : int, default=100
        Logging frequency
    save_every : int, default=500
        Checkpoint saving frequency
    sample_every : int, default=500
        Sample generation frequency
    mixed_precision : str, default="no"
        Mixed precision mode
    gradient_checkpointing : bool, default=True
        Whether to use gradient checkpointing
    """
    # Set up paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    reward_path = Path(reward_model_path)
    if not reward_path.exists():
        raise FileNotFoundError(f"Reward model not found: {reward_path}")
    
    pairs_path = Path(pairs_file)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
    
    # Create dataloader
    dataloader = create_dataloader(
        pairs_file=pairs_file,
        batch_size=batch_size,
    )
    
    # Initialize DSPO fine-tuner
    logger.info("Initializing DSPO fine-tuner...")
    tuner = DSPOFineTuner(
        rank=rank,
        alpha=alpha,
        learning_rate=learning_rate,
        beta=beta,
        mixed_precision=mixed_precision,
        gradient_checkpointing=gradient_checkpointing,
    )
    
    # Load models
    logger.info("Loading models...")
    tuner.load_models(
        reward_model_path=str(reward_path),
        torch_dtype=torch.float16 if mixed_precision == "fp16" else torch.float32,
    )
    
    # Set up optimizer
    tuner.setup_optimizer()
    
    # Sample prompts for evaluation
    sample_prompts = [
        "A serene mountain landscape at sunset",
        "A futuristic city with flying cars",
        "A portrait of a wise old wizard",
        "A cute cat playing in a garden",
        "Abstract art with vibrant colors",
        "A cozy cabin in the woods",
    ]
    
    # Start training
    logger.info(f"Starting DSPO training for {num_steps} steps")
    logger.info(f"Rank: {rank}, Alpha: {tuner.alpha}")
    logger.info(f"Learning rate: {learning_rate}, Beta: {beta}")
    logger.info(f"Batch size: {batch_size}")
    
    tuner.train(
        dataloader=dataloader,
        num_steps=num_steps,
        log_every=log_every,
        save_every=save_every,
        sample_every=sample_every,
        output_dir=str(output_path),
        sample_prompts=sample_prompts,
    )
    
    logger.info("DSPO training completed successfully")


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train DSPO model for preference optimization"
    )
    parser.add_argument(
        "--reward",
        type=str,
        required=True,
        help="Path to trained reward model",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Path to pairs.jsonl file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/dspo_model",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="DoRA rank parameter",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=None,
        help="DoRA alpha parameter (defaults to 4 * rank)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Preference optimization beta parameter",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Logging frequency",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Checkpoint saving frequency",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=500,
        help="Sample generation frequency",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    
    args = parser.parse_args()
    
    # Train model
    train_dspo_model(
        reward_model_path=args.reward,
        pairs_file=args.pairs,
        output_dir=args.output,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        beta=args.beta,
        batch_size=args.batch_size,
        num_steps=args.steps,
        log_every=args.log_every,
        save_every=args.save_every,
        sample_every=args.sample_every,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )


if __name__ == "__main__":
    main()
