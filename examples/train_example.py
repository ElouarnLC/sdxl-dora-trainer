# Example Python Script for Training
# This script shows how to use the trainer programmatically

import logging
from pathlib import Path
from sdxl_dora_trainer import DoRATrainer, TrainingConfig

def main():
    """Example training script."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create training configuration
    config = TrainingConfig(
        # Required parameters
        dataset_path="./my_dataset",
        
        # DoRA parameters
        rank=32,
        alpha=16,
        
        # Training parameters
        learning_rate=5e-5,
        batch_size=1,
        max_train_steps=1000,
        
        # Output settings
        output_dir="./output",
        
        # Memory optimization
        mixed_precision="no",  # Use "no" for stability
        gradient_checkpointing=True,
        use_8bit_adam=True,
        
        # Logging
        report_to="tensorboard",
        project_name="example-training"
    )
    
    # Validate dataset exists
    if not Path(config.dataset_path).exists():
        print(f"Error: Dataset path does not exist: {config.dataset_path}")
        return
    
    # Create trainer
    trainer = DoRATrainer(config)
    
    # Start training
    try:
        print("Starting training...")
        trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
