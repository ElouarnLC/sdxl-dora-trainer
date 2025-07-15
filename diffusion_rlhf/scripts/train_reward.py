#!/usr/bin/env python3
"""
Train multi-head reward model for image preference learning.

Trains a reward model with frozen OpenCLIP backbone and 5 specialized MLP heads
for spatial, iconographic, style, fidelity, and material aspects.
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from dspo.datasets import PairPreferenceDataset
from dspo.multihead_dataset import MultiHeadPairPreferenceDataset
from dspo.reward import MultiHeadReward

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_accelerator(mixed_precision: str = "no") -> Accelerator:
    """
    Set up Accelerator for distributed training.
    
    Parameters
    ----------
    mixed_precision : str
        Mixed precision mode ("no", "fp16", "bf16")
        
    Returns
    -------
    Accelerator
        Configured accelerator
    """
    return Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir="outputs/logs",
    )


def create_dataloaders(
    pairs_file: str,
    batch_size: int = 8,
    val_split: float = 0.2,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Parameters
    ----------
    pairs_file : str
        Path to pairs.jsonl file
    batch_size : int
        Batch size
    val_split : float
        Validation split ratio
    num_workers : int
        Number of dataloader workers
        
    Returns
    -------
    tuple[DataLoader, DataLoader]
        Training and validation dataloaders
    """
    # Create datasets
    train_dataset = PairPreferenceDataset(
        pairs_file=pairs_file,
        split="train",
        val_split=val_split,
    )
    
    val_dataset = PairPreferenceDataset(
        pairs_file=pairs_file,
        split="val",
        val_split=val_split,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def create_multihead_dataloaders(
    ratings_file: str,
    batch_size: int = 8,
    val_split: float = 0.2,
    num_workers: int = 4,
    min_rating_diff: float = 0.5,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders for multi-head training.
    
    Parameters
    ----------
    ratings_file : str
        Path to ratings.csv file with multi-head ratings
    batch_size : int
        Batch size
    val_split : float
        Validation split ratio
    num_workers : int
        Number of dataloader workers
    min_rating_diff : float
        Minimum rating difference to create preference pair
        
    Returns
    -------
    tuple[DataLoader, DataLoader]
        Training and validation dataloaders
    """
    # Create datasets
    train_dataset = MultiHeadPairPreferenceDataset(
        ratings_file=ratings_file,
        split="train",
        val_split=val_split,
        min_rating_diff=min_rating_diff,
    )
    
    val_dataset = MultiHeadPairPreferenceDataset(
        ratings_file=ratings_file,
        split="val",
        val_split=val_split,
        min_rating_diff=min_rating_diff,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Log dataset statistics
    stats = train_dataset.get_dataset_stats()
    logger.info("Dataset statistics:")
    for head, counts in stats.get("head_preferences", {}).items():
        logger.info(f"  {head}: {counts}")
    
    return train_loader, val_loader


def train_epoch(
    model: MultiHeadReward,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
) -> dict[str, float]:
    """
    Train for one epoch.
    
    Parameters
    ----------
    model : MultiHeadReward
        Reward model
    train_loader : DataLoader
        Training dataloader
    optimizer : torch.optim.Optimizer
        Optimizer
    accelerator : Accelerator
        Accelerator instance
    epoch : int
        Current epoch number
        
    Returns
    -------
    dict[str, float]
        Training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        disable=not accelerator.is_local_main_process,
    )
    
    for batch in progress_bar:
        # Forward pass
        loss = model.compute_preference_loss(
            img_pos=batch["img_pos"],
            img_neg=batch["img_neg"],
            labels=batch["label"],
        )
        
        # Backward pass
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return {"train_loss": avg_loss}


def train_multihead_epoch(
    model: MultiHeadReward,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
) -> dict[str, float]:
    """
    Train for one epoch with multi-head loss.
    
    Parameters
    ----------
    model : MultiHeadReward
        Reward model
    train_loader : DataLoader
        Training dataloader
    optimizer : torch.optim.Optimizer
        Optimizer
    accelerator : Accelerator
        Accelerator instance
    epoch : int
        Current epoch number
        
    Returns
    -------
    dict[str, float]
        Training metrics including per-head losses
    """
    model.train()
    total_losses = {}
    num_batches = 0
    
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        disable=not accelerator.is_local_main_process,
    )
    
    for batch in progress_bar:
        # Forward pass with multi-head loss
        losses = model.compute_multihead_preference_loss(
            img_pos=batch["img_pos"],
            img_neg=batch["img_neg"],
            labels=batch["labels"],
        )
        
        # Backward pass
        accelerator.backward(losses["total_loss"])
        optimizer.step()
        optimizer.zero_grad()
        
        # Update metrics
        for key, value in losses.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            total_losses[key] += value.item()
        
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix(
            loss=f"{losses['total_loss'].item():.4f}",
            spatial=f"{losses.get('spatial_loss', torch.tensor(0.0)).item():.3f}",
            style=f"{losses.get('style_loss', torch.tensor(0.0)).item():.3f}",
        )
    
    # Calculate average losses
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}
    return {f"train_{key}": value for key, value in avg_losses.items()}


def validate_epoch(
    model: MultiHeadReward,
    val_loader: DataLoader,
    accelerator: Accelerator,
) -> dict[str, float]:
    """
    Validate for one epoch.
    
    Parameters
    ----------
    model : MultiHeadReward
        Reward model
    val_loader : DataLoader
        Validation dataloader
    accelerator : Accelerator
        Accelerator instance
        
    Returns
    -------
    dict[str, float]
        Validation metrics
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(
            val_loader,
            desc="Validation",
            disable=not accelerator.is_local_main_process,
        ):
            # Forward pass
            loss = model.compute_preference_loss(
                img_pos=batch["img_pos"],
                img_neg=batch["img_neg"],
                labels=batch["label"],
            )
            
            # Get predictions for metrics
            reward_pos = model(batch["img_pos"])
            reward_neg = model(batch["img_neg"])
            
            # Convert to binary predictions
            logits = reward_pos - reward_neg
            predictions = (torch.sigmoid(logits) > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())
            total_loss += loss.item()
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="binary")
    
    return {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
        "val_f1": f1,
    }


def validate_multihead_epoch(
    model: MultiHeadReward,
    val_loader: DataLoader,
    accelerator: Accelerator,
) -> dict[str, float]:
    """Validate for one epoch with multi-head metrics."""
    model.eval()
    total_losses = {}
    head_predictions = {head: [] for head in ["spatial", "icono", "style", "fidelity", "material"]}
    head_labels = {head: [] for head in ["spatial", "icono", "style", "fidelity", "material"]}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", disable=not accelerator.is_local_main_process):
            # Forward pass with multi-head loss
            losses = model.compute_multihead_preference_loss(
                img_pos=batch["img_pos"],
                img_neg=batch["img_neg"],
                labels=batch["labels"],
            )
            
            # Get individual head predictions
            head_outputs_pos = model(batch["img_pos"], return_individual=True)
            head_outputs_neg = model(batch["img_neg"], return_individual=True)
            
            # Process each head
            head_names = ["spatial", "icono", "style", "fidelity", "material"]
            for i, head_name in enumerate(head_names):
                head_logits = head_outputs_pos[head_name] - head_outputs_neg[head_name]
                head_pred = (torch.sigmoid(head_logits) > 0.5).float()
                
                # Only include non-neutral labels (not 0.5)
                head_labels_batch = batch["labels"][:, i]
                valid_mask = head_labels_batch != 0.5
                
                if valid_mask.any():
                    head_predictions[head_name].extend(head_pred[valid_mask].cpu().numpy())
                    head_labels[head_name].extend(head_labels_batch[valid_mask].cpu().numpy())
            
            # Update loss metrics
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()
    
    # Calculate metrics
    num_batches = len(val_loader)
    avg_losses = {f"val_{key}": value / num_batches for key, value in total_losses.items()}
    
    # Calculate per-head metrics
    head_metrics = {}
    for head_name in head_names:
        if head_predictions[head_name]:
            head_acc = accuracy_score(head_labels[head_name], head_predictions[head_name])
            head_f1 = f1_score(head_labels[head_name], head_predictions[head_name], average="binary")
            head_metrics[f"val_{head_name}_accuracy"] = head_acc
            head_metrics[f"val_{head_name}_f1"] = head_f1
    
    return {**avg_losses, **head_metrics}


def train_reward_model(
    pairs_file: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    val_split: float = 0.2,
    mixed_precision: str = "no",
    save_every: int = 2,
) -> None:
    """
    Train the reward model.
    
    Parameters
    ----------
    pairs_file : str
        Path to pairs.jsonl file
    output_dir : str
        Output directory for model checkpoints
    num_epochs : int, default=10
        Number of training epochs
    batch_size : int, default=8
        Batch size
    learning_rate : float, default=1e-4
        Learning rate
    weight_decay : float, default=0.01
        Weight decay
    val_split : float, default=0.2
        Validation split ratio
    mixed_precision : str, default="no"
        Mixed precision mode
    save_every : int, default=2
        Save checkpoint every N epochs
    """
    # Set up paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up accelerator
    accelerator = setup_accelerator(mixed_precision)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        pairs_file=pairs_file,
        batch_size=batch_size,
        val_split=val_split,
    )
    
    # Initialize model
    model = MultiHeadReward()
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Prepare for training
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    logger.info(f"Starting reward model training for {num_epochs} epochs")
    logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    best_f1 = 0.0
    
    for epoch in range(1, num_epochs + 1):
        # Training
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            accelerator=accelerator,
            epoch=epoch,
        )
        
        # Validation
        val_metrics = validate_epoch(
            model=model,
            val_loader=val_loader,
            accelerator=accelerator,
        )
        
        # Combine metrics
        metrics = {**train_metrics, **val_metrics}
        
        # Log metrics
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        # Save best model
        if val_metrics["val_f1"] > best_f1:
            best_f1 = val_metrics["val_f1"]
            if accelerator.is_main_process:
                logger.info(f"New best F1: {best_f1:.4f}, saving model")
                accelerator.unwrap_model(model).save_pretrained(output_path / "best")
        
        # Save checkpoint
        if epoch % save_every == 0 and accelerator.is_main_process:
            accelerator.unwrap_model(model).save_pretrained(
                output_path / f"checkpoint-{epoch}"
            )
    
    # Save final model
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(output_path / "final")
        logger.info(f"Training completed. Best F1: {best_f1:.4f}")
        logger.info(f"Models saved to {output_path}")


def train_multihead_reward_model(
    ratings_file: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    val_split: float = 0.2,
    mixed_precision: str = "no",
    save_every: int = 2,
    min_rating_diff: float = 0.5,
) -> None:
    """
    Train the multi-head reward model with individual head ratings.
    
    Parameters
    ----------
    ratings_file : str
        Path to ratings.csv file with multi-head ratings
    output_dir : str
        Output directory for model checkpoints
    num_epochs : int, default=10
        Number of training epochs
    batch_size : int, default=8
        Batch size
    learning_rate : float, default=1e-4
        Learning rate
    weight_decay : float, default=0.01
        Weight decay
    val_split : float, default=0.2
        Validation split ratio
    mixed_precision : str, default="no"
        Mixed precision mode
    save_every : int, default=2
        Save checkpoint every N epochs
    min_rating_diff : float, default=0.5
        Minimum rating difference to create preference pair
    """
    # Set up paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up accelerator
    accelerator = setup_accelerator(mixed_precision)
    
    # Create dataloaders
    train_loader, val_loader = create_multihead_dataloaders(
        ratings_file=ratings_file,
        batch_size=batch_size,
        val_split=val_split,
        min_rating_diff=min_rating_diff,
    )
    
    # Initialize model
    model = MultiHeadReward()
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Prepare for training
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    logger.info(f"Starting multi-head reward model training for {num_epochs} epochs")
    logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    logger.info(f"Min rating difference: {min_rating_diff}")
    
    best_f1 = 0.0
    
    for epoch in range(1, num_epochs + 1):
        # Training
        train_metrics = train_multihead_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            accelerator=accelerator,
            epoch=epoch,
        )
        
        # Validation
        val_metrics = validate_multihead_epoch(
            model=model,
            val_loader=val_loader,
            accelerator=accelerator,
        )
        
        # Combine metrics
        metrics = {**train_metrics, **val_metrics}
        
        # Log metrics
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        # Save best model based on combined F1
        current_f1 = val_metrics.get("val_combined_f1", 0.0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            if accelerator.is_main_process:
                logger.info(f"New best F1: {best_f1:.4f}, saving model")
                accelerator.unwrap_model(model).save_pretrained(output_path / "best")
        
        # Save checkpoint
        if epoch % save_every == 0 and accelerator.is_main_process:
            accelerator.unwrap_model(model).save_pretrained(
                output_path / f"checkpoint-{epoch}"
            )
    
    # Save final model
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(output_path / "final")
        logger.info(f"Training completed. Best F1: {best_f1:.4f}")
        logger.info(f"Models saved to {output_path}")
        
        # Log final head importance
        head_importance = accelerator.unwrap_model(model).get_head_importance()
        logger.info("Final head importance weights:")
        for head, weight in head_importance.items():
            logger.info(f"  {head}: {weight:.4f}")


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train multi-head reward model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pairs", "multihead"],
        default="pairs",
        help="Training mode: 'pairs' for pairs.jsonl, 'multihead' for ratings.csv",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        help="Path to pairs.jsonl file (required for pairs mode)",
    )
    parser.add_argument(
        "--ratings",
        type=str,
        help="Path to ratings.csv file (required for multihead mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/reward_model",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=2,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--min-rating-diff",
        type=float,
        default=0.5,
        help="Minimum rating difference for multihead mode",
    )
    
    args = parser.parse_args()
    
    # Validate inputs based on mode
    if args.mode == "pairs":
        if not args.pairs:
            raise ValueError("--pairs is required for pairs mode")
        pairs_file = Path(args.pairs)
        if not pairs_file.exists():
            raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
        
        # Train with pairs
        train_reward_model(
            pairs_file=str(pairs_file),
            output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            val_split=args.val_split,
            mixed_precision=args.mixed_precision,
            save_every=args.save_every,
        )
    
    elif args.mode == "multihead":
        if not args.ratings:
            raise ValueError("--ratings is required for multihead mode")
        ratings_file = Path(args.ratings)
        if not ratings_file.exists():
            raise FileNotFoundError(f"Ratings file not found: {ratings_file}")
        
        # Train with multi-head ratings
        train_multihead_reward_model(
            ratings_file=str(ratings_file),
            output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            val_split=args.val_split,
            mixed_precision=args.mixed_precision,
            save_every=args.save_every,
            min_rating_diff=args.min_rating_diff,
        )


if __name__ == "__main__":
    main()