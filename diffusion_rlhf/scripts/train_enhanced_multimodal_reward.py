#!/usr/bin/env python3
"""
Enhanced training script for multimodal reward models with advanced techniques.

This script implements state-of-the-art improvements for reward model training:
1. Advanced data augmentation and mixup
2. Better learning rate scheduling
3. Gradient accumulation and clipping
4. Exponential moving average (EMA)
5. Advanced loss functions and regularization
6. Improved validation metrics
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from dspo.multimodal_datasets import (
    MultimodalPairPreferenceDataset,
    MultimodalMultiHeadPairPreferenceDataset,
)
from dspo.enhanced_multimodal_reward import (
    EnhancedMultimodalMultiHeadReward,
    create_enhanced_training_config,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MixUpAugmentation:
    """
    MixUp data augmentation for preference learning.
    
    Parameters
    ----------
    alpha : float, default=0.2
        MixUp interpolation parameter
    """
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply MixUp augmentation to batch."""
        if self.alpha <= 0:
            return batch
        
        batch_size = batch["img_pos"].size(0)
        lam = np.random.beta(self.alpha, self.alpha, batch_size)
        lam = torch.from_numpy(lam).float().to(batch["img_pos"].device)
        
        # Random permutation for mixing
        indices = torch.randperm(batch_size).to(batch["img_pos"].device)
        
        # Mix images
        lam_expanded = lam.view(-1, 1, 1, 1)
        batch["img_pos"] = lam_expanded * batch["img_pos"] + (1 - lam_expanded) * batch["img_pos"][indices]
        batch["img_neg"] = lam_expanded * batch["img_neg"] + (1 - lam_expanded) * batch["img_neg"][indices]
        
        # Mix labels
        if "label" in batch:
            batch["label"] = lam * batch["label"] + (1 - lam) * batch["label"][indices]
        if "labels" in batch:
            batch["labels"] = lam.unsqueeze(1) * batch["labels"] + (1 - lam.unsqueeze(1)) * batch["labels"][indices]
        
        return batch


class EMA:
    """
    Exponential Moving Average for model parameters.
    
    Parameters
    ----------
    model : nn.Module
        Model to track
    decay : float, default=0.999
        EMA decay rate
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def create_enhanced_optimizer(
    model: nn.Module, 
    learning_rate: float = 5e-5,
    weight_decay: float = 0.02,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """
    Create enhanced optimizer with parameter-specific learning rates.
    
    Parameters
    ----------
    model : nn.Module
        Model to optimize
    learning_rate : float
        Base learning rate
    weight_decay : float
        Weight decay coefficient
    betas : tuple[float, float]
        Adam beta parameters
        
    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer
    """
    # Separate parameters by type for different learning rates
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
    
    # Different learning rates for different parts
    param_groups = [
        {
            "params": backbone_params,
            "lr": learning_rate * 0.1,  # Lower LR for backbone
            "weight_decay": weight_decay,
        },
        {
            "params": head_params,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_groups, betas=betas)
    return optimizer


def train_enhanced_epoch(
    model: EnhancedMultimodalMultiHeadReward,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    accelerator: Accelerator,
    epoch: int,
    config: dict[str, Any],
    ema: EMA = None,
    mixup: MixUpAugmentation = None,
) -> dict[str, float]:
    """
    Enhanced training epoch with advanced techniques.
    
    Parameters
    ----------
    model : EnhancedMultimodalMultiHeadReward
        Enhanced model
    train_loader : DataLoader
        Training dataloader
    optimizer : torch.optim.Optimizer
        Optimizer
    scheduler : Any
        Learning rate scheduler
    accelerator : Accelerator
        Accelerator instance
    epoch : int
        Current epoch
    config : dict[str, Any]
        Training configuration
    ema : EMA, optional
        Exponential moving average
    mixup : MixUpAugmentation, optional
        MixUp augmentation
        
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
    
    # Gradient accumulation
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    
    for step, batch in enumerate(progress_bar):
        # Apply MixUp augmentation
        if mixup is not None and model.training:
            batch = mixup(batch)
        
        # Forward pass with enhanced loss
        loss = model.compute_enhanced_loss(
            img_pos=batch["img_pos"],
            img_neg=batch["img_neg"],
            prompts=batch["prompt"],
            labels=batch.get("label", batch.get("labels")),
        )
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        accelerator.backward(loss)
        
        # Gradient clipping and optimization step
        if (step + 1) % gradient_accumulation_steps == 0:
            if max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update EMA
            if ema is not None:
                ema.update()
        
        # Update metrics
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        # Update progress bar
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix(
            loss=f"{loss.item() * gradient_accumulation_steps:.4f}",
            lr=f"{current_lr:.2e}"
        )
    
    avg_loss = total_loss / num_batches
    return {"train_loss": avg_loss}


def validate_enhanced_epoch(
    model: EnhancedMultimodalMultiHeadReward,
    val_loader: DataLoader,
    accelerator: Accelerator,
    ema: EMA = None,
) -> dict[str, float]:
    """
    Enhanced validation with comprehensive metrics.
    
    Parameters
    ----------
    model : EnhancedMultimodalMultiHeadReward
        Enhanced model
    val_loader : DataLoader
        Validation dataloader
    accelerator : Accelerator
        Accelerator instance
    ema : EMA, optional
        Use EMA weights for validation
        
    Returns
    -------
    dict[str, float]
        Validation metrics
    """
    # Use EMA weights for validation if available
    if ema is not None:
        ema.apply_shadow()
    
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(
            val_loader,
            desc="Validation",
            disable=not accelerator.is_local_main_process,
        ):
            # Forward pass
            loss = model.compute_enhanced_loss(
                img_pos=batch["img_pos"],
                img_neg=batch["img_neg"],
                prompts=batch["prompt"],
                labels=batch.get("label", batch.get("labels")),
            )
            
            # Get enhanced metrics
            metrics = model.get_enhanced_metrics(
                img_pos=batch["img_pos"],
                img_neg=batch["img_neg"],
                prompts=batch["prompt"],
                labels=batch.get("label", batch.get("labels")),
            )
            
            # Get predictions for comprehensive metrics
            reward_pos = model(batch["img_pos"], batch["prompt"])
            reward_neg = model(batch["img_neg"], batch["prompt"])
            
            logits = reward_pos - reward_neg
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch.get("label", batch.get("labels")).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_loss += loss.item()
    
    # Restore original weights if using EMA
    if ema is not None:
        ema.restore()
    
    # Calculate comprehensive metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="binary")
    
    # AUC-ROC for better evaluation
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_roc = 0.5  # Default if only one class
    
    # Confidence and calibration metrics
    confidences = np.abs(np.array(all_probs) - 0.5) * 2
    avg_confidence = np.mean(confidences)
    
    # Accuracy at different confidence thresholds
    high_conf_mask = confidences > 0.8
    if np.sum(high_conf_mask) > 0:
        high_conf_accuracy = accuracy_score(
            np.array(all_labels)[high_conf_mask],
            np.array(all_predictions)[high_conf_mask]
        )
    else:
        high_conf_accuracy = 0.0
    
    return {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
        "val_f1": f1,
        "val_auc_roc": auc_roc,
        "val_confidence": avg_confidence,
        "val_high_conf_accuracy": high_conf_accuracy,
    }


def train_enhanced_multimodal_reward_model(
    data_file: str,
    prompts_file: str,
    output_dir: str,
    config: dict[str, Any],
) -> None:
    """
    Train enhanced multimodal reward model with advanced techniques.
    
    Parameters
    ----------
    data_file : str
        Path to data file (pairs.jsonl or ratings.csv)
    prompts_file : str
        Path to prompts.csv file
    output_dir : str
        Output directory for model checkpoints
    config : dict[str, Any]
        Training configuration
    """
    # Set up paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up accelerator
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "no"),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        log_with="tensorboard",
        project_dir=output_path / "logs",
    )
    
    # Determine dataset type
    data_path = Path(data_file)
    if data_path.suffix == ".jsonl":
        # Pairs dataset
        train_dataset = MultimodalPairPreferenceDataset(
            pairs_file=data_file,
            prompts_file=prompts_file,
            split="train",
            val_split=config.get("val_split", 0.2),
        )
        
        val_dataset = MultimodalPairPreferenceDataset(
            pairs_file=data_file,
            prompts_file=prompts_file,
            split="val",
            val_split=config.get("val_split", 0.2),
        )
    else:
        # Multi-head ratings dataset
        train_dataset = MultimodalMultiHeadPairPreferenceDataset(
            ratings_file=data_file,
            prompts_file=prompts_file,
            split="train",
            val_split=config.get("val_split", 0.2),
            min_rating_diff=config.get("min_rating_diff", 0.5),
        )
        
        val_dataset = MultimodalMultiHeadPairPreferenceDataset(
            ratings_file=data_file,
            prompts_file=prompts_file,
            split="val",
            val_split=config.get("val_split", 0.2),
            min_rating_diff=config.get("min_rating_diff", 0.5),
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
    
    # Initialize enhanced model
    model = EnhancedMultimodalMultiHeadReward(
        model_name=config.get("model_name", "ViT-L-14-336"),
        fusion_method=config.get("fusion_method", "attention"),
        loss_type=config.get("loss_type", "focal"),
        hidden_dims=config.get("hidden_dims", [512, 256]),
        dropout=config.get("dropout", 0.15),
        temperature=config.get("temperature", 0.07),
    )
    
    # Set up enhanced optimizer
    optimizer = create_enhanced_optimizer(
        model,
        learning_rate=config.get("learning_rate", 5e-5),
        weight_decay=config.get("weight_decay", 0.02),
    )
    
    # Set up scheduler
    num_training_steps = len(train_loader) * config.get("num_epochs", 10)
    warmup_steps = config.get("warmup_steps", num_training_steps // 10)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Set up augmentation and EMA
    mixup = MixUpAugmentation(alpha=config.get("mixup_alpha", 0.2))
    ema = EMA(model, decay=config.get("ema_decay", 0.999))
    
    # Prepare for training
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    logger.info("Enhanced multimodal reward model training started")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Configuration: {config}")
    
    best_f1 = 0.0
    best_auc = 0.0
    patience = config.get("patience", 5)
    patience_counter = 0
    
    for epoch in range(1, config.get("num_epochs", 10) + 1):
        # Training
        train_metrics = train_enhanced_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            epoch=epoch,
            config=config,
            ema=ema,
            mixup=mixup,
        )
        
        # Validation
        val_metrics = validate_enhanced_epoch(
            model=model,
            val_loader=val_loader,
            accelerator=accelerator,
            ema=ema,
        )
        
        # Combine metrics
        metrics = {**train_metrics, **val_metrics}
        
        # Log metrics
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        # Save best model based on F1 score
        current_f1 = val_metrics["val_f1"]
        current_auc = val_metrics["val_auc_roc"]
        
        # Use combined metric for best model selection
        combined_metric = (current_f1 + current_auc) / 2
        best_combined = (best_f1 + best_auc) / 2
        
        if combined_metric > best_combined:
            best_f1 = current_f1
            best_auc = current_auc
            patience_counter = 0
            
            if accelerator.is_main_process:
                logger.info(f"New best model - F1: {best_f1:.4f}, AUC: {best_auc:.4f}")
                # Save with EMA weights
                if ema:
                    ema.apply_shadow()
                accelerator.unwrap_model(model).save_pretrained(output_path / "best")
                if ema:
                    ema.restore()
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping after {epoch} epochs")
            break
        
        # Save checkpoint
        if epoch % config.get("save_every", 5) == 0 and accelerator.is_main_process:
            if ema:
                ema.apply_shadow()
            accelerator.unwrap_model(model).save_pretrained(
                output_path / f"checkpoint-{epoch}"
            )
            if ema:
                ema.restore()
    
    # Save final model
    if accelerator.is_main_process:
        if ema:
            ema.apply_shadow()
        accelerator.unwrap_model(model).save_pretrained(output_path / "final")
        logger.info(f"Training completed. Best F1: {best_f1:.4f}, Best AUC: {best_auc:.4f}")
        logger.info(f"Models saved to {output_path}")


def main() -> None:
    """Main training function with enhanced configuration."""
    parser = argparse.ArgumentParser(
        description="Enhanced multimodal reward model training"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data file (pairs.jsonl or ratings.csv)",
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
        default="outputs/enhanced_multimodal_reward",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="enhanced",
        choices=["enhanced", "baseline"],
        help="Training configuration preset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == "enhanced":
        config = create_enhanced_training_config()
    else:
        config = {
            "model_name": "ViT-L-14",
            "fusion_method": "concat",
            "loss_type": "bce",
            "hidden_dims": [512, 256],
            "dropout": 0.1,
            "learning_rate": 1e-4,
            "batch_size": 8,
            "num_epochs": 10,
        }
    
    # Override with command line arguments
    config.update({
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    })
    
    # Validate inputs
    data_file = Path(args.data)
    prompts_file = Path(args.prompts)
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    # Start training
    train_enhanced_multimodal_reward_model(
        data_file=str(data_file),
        prompts_file=str(prompts_file),
        output_dir=args.output,
        config=config,
    )


if __name__ == "__main__":
    main()
