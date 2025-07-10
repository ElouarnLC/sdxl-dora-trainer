#!/usr/bin/env python3
"""
Optuna hyperparameter sweep for DSPO optimization.

Searches optimal rank and learning rate parameters for DSPO fine-tuning
using Optuna optimization framework.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import optuna
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dspo.datasets import PairPreferenceDataset
from dspo.reward import MultiHeadReward
from dspo.tuner import DSPOFineTuner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_val_dataloader(
    pairs_file: str,
    batch_size: int = 8,
) -> DataLoader:
    """
    Create validation dataloader for evaluation.
    
    Parameters
    ----------
    pairs_file : str
        Path to pairs.jsonl file
    batch_size : int
        Batch size
        
    Returns
    -------
    DataLoader
        Validation dataloader
    """
    dataset = PairPreferenceDataset(
        pairs_file=pairs_file,
        split="val",
        image_size=1024,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    return dataloader


def evaluate_dspo_model(
    tuner: DSPOFineTuner,
    val_loader: DataLoader,
) -> dict[str, float]:
    """
    Evaluate DSPO model on validation set.
    
    Parameters
    ----------
    tuner : DSPOFineTuner
        Trained DSPO fine-tuner
    val_loader : DataLoader
        Validation dataloader
        
    Returns
    -------
    dict[str, float]
        Evaluation metrics
    """
    tuner.unet.eval()
    tuner.reward_model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Get reward scores for positive and negative images
            reward_pos = tuner.reward_model(batch["img_pos"])
            reward_neg = tuner.reward_model(batch["img_neg"])
            
            # Convert to binary predictions
            logits = reward_pos - reward_neg
            predictions = (torch.sigmoid(logits) > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())
    
    # Calculate F1 score
    f1 = f1_score(all_labels, all_predictions, average="binary")
    
    return {"f1_score": f1}


def objective(
    trial: optuna.Trial,
    reward_model_path: str,
    pairs_file: str,
    output_dir: str,
    num_steps: int = 500,
) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    reward_model_path : str
        Path to trained reward model
    pairs_file : str
        Path to pairs.jsonl file
    output_dir : str
        Output directory for trial results
    num_steps : int, default=500
        Number of training steps per trial
        
    Returns
    -------
    float
        F1 score to maximize
    """
    # Suggest hyperparameters
    rank = trial.suggest_categorical("rank", [4, 8, 16])
    learning_rate = trial.suggest_float("learning_rate", 2e-5, 2e-4, log=True)
    
    # Create trial-specific output directory
    trial_output = Path(output_dir) / f"trial_{trial.number}"
    trial_output.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Trial {trial.number}: rank={rank}, lr={learning_rate}")
    
    try:
        # Create training dataloader
        train_dataset = PairPreferenceDataset(
            pairs_file=pairs_file,
            split="train",
            image_size=1024,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,  # Smaller batch size for faster trials
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        
        # Create validation dataloader
        val_loader = create_val_dataloader(pairs_file, batch_size=4)
        
        # Initialize DSPO fine-tuner
        tuner = DSPOFineTuner(
            rank=rank,
            alpha=4 * rank,  # Standard alpha setting
            learning_rate=learning_rate,
            beta=0.1,  # Fixed beta
            mixed_precision="no",  # For stability
            gradient_checkpointing=True,
        )
        
        # Load models
        tuner.load_models(
            reward_model_path=reward_model_path,
            torch_dtype=torch.float32,  # Full precision for stability
        )
        
        # Set up optimizer
        tuner.setup_optimizer()
        
        # Quick training loop
        tuner.unet, tuner.optimizer, train_loader = tuner.accelerator.prepare(
            tuner.unet, tuner.optimizer, train_loader
        )
        
        # Training loop
        step = 0
        for batch in train_loader:
            if step >= num_steps:
                break
            
            # Training step
            metrics = tuner.train_step(batch)
            step += 1
            
            # Report intermediate value for pruning
            if step % 100 == 0:
                # Quick evaluation
                eval_metrics = evaluate_dspo_model(tuner, val_loader)
                trial.report(eval_metrics["f1_score"], step)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # Final evaluation
        eval_metrics = evaluate_dspo_model(tuner, val_loader)
        f1_score = eval_metrics["f1_score"]
        
        logger.info(f"Trial {trial.number} completed: F1 = {f1_score:.4f}")
        
        return f1_score
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return 0.0  # Return worst possible score
    
    finally:
        # Clean up GPU memory
        if "tuner" in locals():
            del tuner
        torch.cuda.empty_cache()


def run_hyperparameter_sweep(
    reward_model_path: str,
    pairs_file: str,
    output_dir: str,
    n_trials: int = 20,
    timeout: int | None = None,
    study_name: str = "dspo_optimization",
) -> None:
    """
    Run Optuna hyperparameter sweep.
    
    Parameters
    ----------
    reward_model_path : str
        Path to trained reward model
    pairs_file : str
        Path to pairs.jsonl file
    output_dir : str
        Output directory for results
    n_trials : int, default=20
        Number of trials to run
    timeout : int | None, default=None
        Timeout in seconds (None for no timeout)
    study_name : str, default="dspo_optimization"
        Study name for Optuna
    """
    # Set up output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=100,
            interval_steps=100,
        ),
    )
    
    logger.info(f"Starting hyperparameter sweep with {n_trials} trials")
    logger.info(f"Search space: rank ∈ {[4, 8, 16]}, lr ∈ [2e-5, 2e-4]")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial=trial,
            reward_model_path=reward_model_path,
            pairs_file=pairs_file,
            output_dir=str(output_path),
        ),
        n_trials=n_trials,
        timeout=timeout,
    )
    
    # Save results
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info("Hyperparameter sweep completed!")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best F1 score: {best_value:.4f}")
    
    # Save study results
    study_file = output_path / "study_results.pkl"
    with open(study_file, "wb") as f:
        import pickle
        pickle.dump(study, f)
    
    # Save best parameters
    import json
    with open(output_path / "best_params.json", "w") as f:
        json.dump({
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(study.trials),
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def main() -> None:
    """Main sweep function."""
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter sweep for DSPO"
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
        default="outputs/sweep",
        help="Output directory for sweep results",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds (None for no timeout)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="dspo_optimization",
        help="Study name for Optuna",
    )
    parser.add_argument(
        "--steps-per-trial",
        type=int,
        default=500,
        help="Number of training steps per trial",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    reward_path = Path(args.reward)
    if not reward_path.exists():
        raise FileNotFoundError(f"Reward model not found: {reward_path}")
    
    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
    
    # Run sweep with custom objective
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=100,
            interval_steps=100,
        ),
    )
    
    logger.info(f"Starting hyperparameter sweep with {args.n_trials} trials")
    
    # Custom objective with trial steps
    def custom_objective(trial: optuna.Trial) -> float:
        return objective(
            trial=trial,
            reward_model_path=str(reward_path),
            pairs_file=str(pairs_path),
            output_dir=args.output,
            num_steps=args.steps_per_trial,
        )
    
    # Run optimization
    study.optimize(
        custom_objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info("Hyperparameter sweep completed!")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best F1 score: {best_value:.4f}")
    
    # Save results
    import json
    with open(output_path / "best_params.json", "w") as f:
        json.dump({
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(study.trials),
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
