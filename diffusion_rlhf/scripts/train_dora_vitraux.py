#!/usr/bin/env python3
"""
DoRA training on real vitraux dataset with DSPO evaluation.

This script:
1. Trains DoRA on real vitraux images (supervised learning on image-text pairs)
2. Evaluates model iterations using DSPO - generates preference pairs and 
   evaluates with reward model
3. Selects best model based on DSPO preference scores

The training uses your real vitraux dataset: datasets/vitraux/*.jpg + *.txt files
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add the diffusion_rlhf directory to path for imports
current_dir = Path(__file__).parent.parent  # diffusion_rlhf directory
sys.path.append(str(current_dir))

# Import existing infrastructure
from diffusers import StableDiffusionXLPipeline
from dspo.trained_reward_model import load_reward_model
from dspo.tuner import DSPOFineTuner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VitrauxDataset(Dataset):
    """Dataset for training on real vitraux images with text annotations."""
    
    def __init__(
        self,
        dataset_path: str,
        transform: transforms.Compose = None
    ):
        """
        Initialize vitraux dataset.
        
        Parameters
        ----------
        dataset_path : str
            Path to vitraux dataset folder
        transform : transforms.Compose, optional
            Image transformations
        """
        self.dataset_path = Path(dataset_path)
        
        # Find all image-text pairs
        self.image_files = list(self.dataset_path.glob("*.jpg"))
        self.data_pairs = []
        
        for img_file in self.image_files:
            # Find corresponding text file
            txt_file = img_file.with_suffix('.txt')
            if txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:  # Only include if text is not empty
                        self.data_pairs.append({
                            'image_path': img_file,
                            'text': text
                        })
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # [-1, 1] for diffusion
            ])
        else:
            self.transform = transform
            
        logger.info(f"Loaded {len(self.data_pairs)} vitraux image-text pairs")
        
    def __len__(self) -> int:
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair = self.data_pairs[idx]
        
        # Load image
        image = Image.open(pair['image_path']).convert('RGB')
        image_tensor = self.transform(image)
        
        return {
            'image': image_tensor,
            'prompt': pair['text'],
            'image_path': str(pair['image_path'])
        }


class DoRATrainer:
    """DoRA trainer for vitraux dataset with supervised learning."""
    
    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        rank: int = 16,
        alpha: int = None,
        learning_rate: float = 5e-5,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.rank = rank
        self.alpha = alpha if alpha is not None else 4 * rank
        self.learning_rate = learning_rate
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize components
        self.pipeline = None
        self.unet = None
        self.optimizer = None
        self.scheduler = None
        
        logger.info(f"Initialized DoRATrainer on {self.device}")
    
    def setup_models(self):
        """Set up SDXL pipeline and DoRA."""
        logger.info("Loading SDXL pipeline...")
        
        # Load pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True,
        )
        self.pipeline = self.pipeline.to(self.device)
        
        # Get UNet and scheduler
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler
        
        # Enable gradient checkpointing
        self.unet.enable_gradient_checkpointing()
        
        # Set up DoRA using DSPOFineTuner's setup
        tuner = DSPOFineTuner(
            model_name=self.model_name,
            rank=self.rank,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            device=self.device
        )
        tuner.unet = self.unet
        tuner._setup_dora()
        self.unet = tuner.unet
        
        # Set up optimizer
        trainable_params = [p for p in self.unet.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        logger.info("Models set up successfully")
    
    def compute_diffusion_loss(
        self, 
        images: torch.Tensor, 
        prompts: list[str]
    ) -> torch.Tensor:
        """
        Compute supervised diffusion loss for image-text pairs.
        
        Parameters
        ----------
        images : torch.Tensor
            Ground truth images [B, C, H, W]
        prompts : list[str]
            Text prompts
            
        Returns
        -------
        torch.Tensor
            Diffusion loss
        """
        batch_size = images.shape[0]
        
        # Encode latents
        with torch.no_grad():
            # Ensure images are in correct range for VAE
            images = torch.clamp(images, -1.0, 1.0)
            latents = self.pipeline.vae.encode(images.float()).latent_dist.sample()
            latents = latents * self.pipeline.vae.config.scaling_factor
            latents = latents.to(images.dtype)

        # Encode text properly for SDXL (using both text encoders)
        with torch.no_grad():
            # Tokenize prompts for both text encoders
            tokens_1 = self.pipeline.tokenizer(
                prompts,
                max_length=self.pipeline.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)

            tokens_2 = self.pipeline.tokenizer_2(
                prompts,
                max_length=self.pipeline.tokenizer_2.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)

            # Encode with both text encoders
            encoder_output_1 = self.pipeline.text_encoder(tokens_1)
            prompt_embeds_1 = encoder_output_1.last_hidden_state

            encoder_output_2 = self.pipeline.text_encoder_2(tokens_2)
            prompt_embeds_2 = encoder_output_2.last_hidden_state
            pooled_prompt_embeds = encoder_output_2.pooler_output

            # Concatenate text embeddings for SDXL
            seq_len = min(prompt_embeds_1.shape[1], prompt_embeds_2.shape[1])
            prompt_embeds_1 = prompt_embeds_1[:, :seq_len, :]
            prompt_embeds_2 = prompt_embeds_2[:, :seq_len, :]
            encoder_hidden_states = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        )

        # Add noise
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Create SDXL conditioning
        original_size = torch.tensor([[images.shape[-2], images.shape[-1]]] * batch_size).to(self.device)
        crops_coords_top_left = torch.tensor([[0, 0]] * batch_size).to(self.device)
        target_size = torch.tensor([[images.shape[-2], images.shape[-1]]] * batch_size).to(self.device)

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": torch.cat([original_size, crops_coords_top_left, target_size], dim=1).float()
        }

        # Predict noise with SDXL UNet
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs
        ).sample
        
        # Compute MSE loss
        loss = F.mse_loss(noise_pred, noise, reduction='mean')
        
        return loss
    
    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Perform single training step."""
        self.optimizer.zero_grad()
        
        images = batch['image'].to(self.device)
        prompts = batch['prompt']
        
        # Compute loss
        loss = self.compute_diffusion_loss(images, prompts)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def save_checkpoint(self, save_path: Path, step: int):
        """Save training checkpoint."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save DoRA weights
        self.unet.save_pretrained(save_path)
        
        # Save optimizer state
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'model_name': self.model_name,
            'rank': self.rank,
            'alpha': self.alpha,
        }, save_path / 'training_state.pt')
        
        logger.info(f"Checkpoint saved to {save_path}")


def evaluate_with_dspo(
    checkpoint_path: str,
    reward_model_path: str,
    prompts: list[str],
    num_pairs: int = 20,
    device: str = "cuda"
) -> dict[str, float]:
    """
    Evaluate checkpoint using DSPO methodology.
    
    Generates preference pairs with different parameters and evaluates
    using the reward model to compute DSPO-style preference scores.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to DoRA checkpoint
    reward_model_path : str
        Path to reward model
    prompts : list[str]
        Prompts for evaluation
    num_pairs : int
        Number of preference pairs to generate
    device : str
        Device to use
        
    Returns
    -------
    dict[str, float]
        DSPO evaluation metrics
    """
    logger.info(f"🔍 DSPO Evaluation: {checkpoint_path}")
    
    # Load pipeline with checkpoint
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    # Load DoRA weights
    from peft import PeftModel
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, checkpoint_path)
    pipeline = pipeline.to(device)
    
    # Load reward model
    reward_model = load_reward_model(reward_model_path)
    
    # DSPO evaluation: generate preference pairs with different parameters
    total_preference_score = 0.0
    preference_accuracy = 0.0
    dspo_loss_sum = 0.0
    
    # Use different CFG scales to create preference pairs
    cfg_low = 7.5
    cfg_high = 10.0
    
    for i in range(num_pairs):
        prompt = np.random.choice(prompts)
        
        # Generate preference pair with different CFG scales
        img_preferred = pipeline(
            prompt=prompt,
            guidance_scale=cfg_high,  # Higher CFG typically better
            num_inference_steps=30,
            generator=torch.Generator().manual_seed(i)
        ).images[0]
        
        img_dispreferred = pipeline(
            prompt=prompt,
            guidance_scale=cfg_low,   # Lower CFG typically worse
            num_inference_steps=30,
            generator=torch.Generator().manual_seed(i + 1000)
        ).images[0]
        
        # Convert for reward model
        def pil_to_reward_tensor(image):
            transform = transforms.Compose([
                transforms.Resize((336, 336)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
            return transform(image)
        
        # Get reward scores
        comparison = reward_model.compare_images(
            pil_to_reward_tensor(img_preferred),
            pil_to_reward_tensor(img_dispreferred),
            prompt
        )
        
        reward_preferred = comparison['reward_a']
        reward_dispreferred = comparison['reward_b']
        
        # Compute DSPO-style metrics
        preference_diff = reward_preferred - reward_dispreferred
        total_preference_score += preference_diff
        
        # Check if preference is correct (higher CFG should be better)
        if preference_diff > 0:
            preference_accuracy += 1.0
        
        # Compute DSPO loss: -log(sigmoid(beta * (r_preferred - r_dispreferred)))
        beta = 0.1  # DSPO hyperparameter
        dspo_loss = -torch.log(torch.sigmoid(beta * preference_diff)).item()
        dspo_loss_sum += dspo_loss
    
    # Average metrics
    avg_preference_score = total_preference_score / num_pairs
    preference_acc = preference_accuracy / num_pairs
    avg_dspo_loss = dspo_loss_sum / num_pairs
    
    # Cleanup
    del pipeline, reward_model
    torch.cuda.empty_cache()
    
    metrics = {
        'dspo_preference_score': avg_preference_score,
        'dspo_accuracy': preference_acc,
        'dspo_loss': avg_dspo_loss,
        'num_evaluated': num_pairs
    }
    
    logger.info(f"📊 DSPO Results: score={avg_preference_score:.4f}, "
                f"acc={preference_acc:.4f}, loss={avg_dspo_loss:.4f}")
    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="DoRA training on vitraux dataset with DSPO evaluation"
    )
    parser.add_argument(
        "--dataset-path", 
        type=str, 
        default="datasets/vitraux", 
        help="Path to vitraux dataset"
    )
    parser.add_argument(
        "--reward-model", 
        type=str, 
        required=True, 
        help="Reward model path"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs/dora_vitraux", 
        help="Output directory"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=2, 
        help="Batch size"
    )
    parser.add_argument(
        "--num-steps", 
        type=int, 
        default=2000, 
        help="Training steps"
    )
    parser.add_argument(
        "--rank", 
        type=int, 
        default=16, 
        help="DoRA rank"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=5e-5, 
        help="Learning rate"
    )
    parser.add_argument(
        "--eval-every", 
        type=int, 
        default=500, 
        help="DSPO evaluation frequency"
    )
    parser.add_argument(
        "--save-every", 
        type=int, 
        default=500, 
        help="Save frequency"
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 DoRA Training on Vitraux Dataset")
    logger.info("=" * 50)
    logger.info(f"📁 Dataset: {args.dataset_path}")
    logger.info(f"🏆 Reward model: {args.reward_model}")
    logger.info(f"💾 Output: {args.output_dir}")
    logger.info(f"📊 DoRA: rank={args.rank}, lr={args.lr}")
    
    # Set up paths
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load vitraux dataset
    dataset = VitrauxDataset(dataset_path)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Extract unique prompts for evaluation
    eval_prompts = [pair['text'] for pair in dataset.data_pairs[:50]]
    
    # Initialize trainer
    trainer = DoRATrainer(
        rank=args.rank,
        learning_rate=args.lr
    )
    trainer.setup_models()
    
    # Training loop
    logger.info(f"🏋️ Starting training for {args.num_steps} steps")
    
    step = 0
    running_loss = 0.0
    best_dspo_score = -float('inf')
    
    with tqdm(total=args.num_steps, desc="Training") as pbar:
        while step < args.num_steps:
            for batch in dataloader:
                if step >= args.num_steps:
                    break
                
                # Training step
                metrics = trainer.train_step(batch)
                running_loss += metrics['loss']
                
                step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{metrics['loss']:.4f}")
                
                # Logging
                if step % 100 == 0:
                    avg_loss = running_loss / 100
                    logger.info(f"Step {step}: loss={avg_loss:.4f}")
                    running_loss = 0.0
                
                # Save checkpoint
                if step % args.save_every == 0:
                    checkpoint_path = output_dir / f"checkpoint-{step}"
                    trainer.save_checkpoint(checkpoint_path, step)
                    
                    # DSPO evaluation
                    if step % args.eval_every == 0:
                        dspo_metrics = evaluate_with_dspo(
                            str(checkpoint_path),
                            args.reward_model,
                            eval_prompts,
                            num_pairs=20
                        )
                        
                        # Save best model based on DSPO preference score
                        dspo_score = dspo_metrics['dspo_preference_score']
                        if dspo_score > best_dspo_score:
                            best_dspo_score = dspo_score
                            best_path = output_dir / "best_model"
                            trainer.save_checkpoint(best_path, step)
                            logger.info(f"🏆 New best model! "
                                        f"DSPO score: {best_dspo_score:.4f}")
                        
                        # Save evaluation results
                        eval_results = {
                            'step': step,
                            'loss': avg_loss,
                            **dspo_metrics
                        }
                        
                        results_file = output_dir / "dspo_evaluation_results.csv"
                        eval_df = pd.DataFrame([eval_results])
                        
                        if results_file.exists():
                            existing_df = pd.read_csv(results_file)
                            eval_df = pd.concat([existing_df, eval_df], 
                                                ignore_index=True)
                        
                        eval_df.to_csv(results_file, index=False)
    
    # Final checkpoint
    final_path = output_dir / "final_model"
    trainer.save_checkpoint(final_path, step)
    
    # Final DSPO evaluation
    final_metrics = evaluate_with_dspo(
        str(final_path),
        args.reward_model, 
        eval_prompts,
        num_pairs=50
    )
    
    logger.info("🎉 Training completed!")
    logger.info(f"🏆 Best DSPO score: {best_dspo_score:.4f}")
    logger.info(f"📊 Final DSPO score: {final_metrics['dspo_preference_score']:.4f}")
    logger.info(f"💾 Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
