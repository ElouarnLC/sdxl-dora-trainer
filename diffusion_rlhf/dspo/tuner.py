"""
DSPO (Diffusion Stable Preference Optimization) fine-tuner for SDXL models.

This module implements the main DSPOFineTuner class that fine-tunes SDXL
models using preference optimization with DoRA adapters.
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm

from .reward import MultiHeadReward

logger = logging.getLogger(__name__)


class DSPOFineTuner:
    """
    DSPO fine-tuner for SDXL models using preference optimization.
    
    Implements preference optimization loss: -log σ(r(x⁺) - r(x⁻))
    where r is the reward model and σ is the sigmoid function.
    
    Parameters
    ----------
    model_name : str, default="stabilityai/stable-diffusion-xl-base-1.0"
        Base SDXL model name
    rank : int, default=8
        DoRA rank parameter
    alpha : int | None, default=None
        DoRA alpha parameter (defaults to 4 * rank)
    learning_rate : float, default=5e-5
        Learning rate for optimization
    beta : float, default=0.1
        Preference optimization beta parameter
    device : str, default="auto"
        Device to use ("auto", "cuda", "cpu")
    mixed_precision : str, default="no"
        Mixed precision mode ("no", "fp16", "bf16")
    gradient_checkpointing : bool, default=True
        Whether to use gradient checkpointing
    """
    
    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        rank: int = 8,
        alpha: int | None = None,
        learning_rate: float = 5e-5,
        beta: float = 0.1,
        device: str = "auto",
        mixed_precision: str = "no",
        gradient_checkpointing: bool = True,
    ) -> None:
        self.model_name = model_name
        self.rank = rank
        self.alpha = alpha if alpha is not None else 4 * rank
        self.learning_rate = learning_rate
        self.beta = beta
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        
        # Initialize components
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        self.pipeline = None
        self.unet = None
        self.reward_model = None
        self.optimizer = None
        
        logger.info(f"Initialized DSPOFineTuner with rank={rank}, alpha={alpha}")
    
    def load_models(
        self,
        reward_model_path: str,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        """
        Load SDXL pipeline and reward model.
        
        Parameters
        ----------
        reward_model_path : str
            Path to trained reward model
        torch_dtype : torch.dtype, default=torch.float16
            Torch dtype for models
        """
        logger.info("Loading SDXL pipeline...")
        
        # Load base pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
        
        # Extract UNet for fine-tuning
        self.unet = self.pipeline.unet
        
        # Enable gradient checkpointing if requested
        if self.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
        
        # Set up DoRA
        self._setup_dora()
        
        # Load reward model
        logger.info(f"Loading reward model from {reward_model_path}")
        self.reward_model = MultiHeadReward.from_pretrained(reward_model_path)
        self.reward_model.eval()
        
        # Freeze reward model
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        logger.info("Models loaded successfully")
    
    def _setup_dora(self) -> None:
        """Set up DoRA configuration for UNet."""
        # Define target modules for DoRA (attention layers)
        target_modules = [
            "to_k",
            "to_q", 
            "to_v",
            "to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
        ]
        
        # Create DoRA config
        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="DIFFUSION",
            use_dora=True,  # Enable DoRA
        )
        
        # Apply DoRA to UNet
        self.unet = get_peft_model(self.unet, lora_config)
        
        # Enable training mode for DoRA parameters
        self.unet.train()
        
        # Count trainable parameters
        trainable_params = sum(
            p.numel() for p in self.unet.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.unet.parameters())
        
        logger.info(f"DoRA parameters: {trainable_params:,} / {total_params:,}")
        logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    def setup_optimizer(self) -> None:
        """Set up optimizer for training."""
        # Only optimize DoRA parameters
        trainable_params = [p for p in self.unet.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8,
        )
        
        logger.info(f"Optimizer configured with lr={self.learning_rate}")
    
    def compute_dspo_loss(
        self,
        prompts: list[str],
        img_pos: torch.Tensor,
        img_neg: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DSPO preference optimization loss.
        
        Parameters
        ----------
        prompts : list[str]
            Text prompts
        img_pos : torch.Tensor
            Positive images [B, C, H, W]
        img_neg : torch.Tensor
            Negative images [B, C, H, W]
        timesteps : torch.Tensor
            Diffusion timesteps [B]
        noise : torch.Tensor
            Noise for diffusion [B, C, H, W]
            
        Returns
        -------
        torch.Tensor
            DSPO loss scalar
        """
        batch_size = len(prompts)
        
        # Encode prompts
        prompt_embeds, _ = self.pipeline._encode_prompt(
            prompts,
            device=self.unet.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        
        # Add noise to images
        noisy_pos = self.pipeline.scheduler.add_noise(img_pos, noise, timesteps)
        noisy_neg = self.pipeline.scheduler.add_noise(img_neg, noise, timesteps)
        
        # Predict noise with UNet
        noise_pred_pos = self.unet(
            noisy_pos,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        
        noise_pred_neg = self.unet(
            noisy_neg,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        
        # Decode predictions to images (simplified)
        # In practice, this would involve the full denoising process
        # Here we use a simplified approximation
        pred_img_pos = img_pos - 0.1 * (noise_pred_pos - noise)
        pred_img_neg = img_neg - 0.1 * (noise_pred_neg - noise)
        
        # Get reward scores
        with torch.no_grad():
            reward_pos = self.reward_model(pred_img_pos)
            reward_neg = self.reward_model(pred_img_neg)
        
        # Compute preference logits
        logits = self.beta * (reward_pos - reward_neg)
        
        # DSPO loss: -log σ(r(x⁺) - r(x⁻))
        loss = -torch.log(torch.sigmoid(logits.squeeze())).mean()
        
        return loss
    
    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        Perform single training step.
        
        Parameters
        ----------
        batch : dict[str, Any]
            Training batch containing prompts and images
            
        Returns
        -------
        dict[str, float]
            Training metrics
        """
        self.optimizer.zero_grad()
        
        # Extract batch data
        prompts = batch["prompt"]
        img_pos = batch["img_pos"]
        img_neg = batch["img_neg"]
        
        batch_size = len(prompts)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.pipeline.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=img_pos.device,
        )
        
        # Sample noise
        noise = torch.randn_like(img_pos)
        
        # Compute loss
        loss = self.compute_dspo_loss(prompts, img_pos, img_neg, timesteps, noise)
        
        # Backward pass
        self.accelerator.backward(loss)
        
        # Gradient clipping
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
        
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }
    
    def train(
        self,
        dataloader: DataLoader,
        num_steps: int = 2000,
        log_every: int = 100,
        save_every: int = 500,
        sample_every: int = 500,
        output_dir: str = "outputs/dspo_model",
        sample_prompts: list[str] | None = None,
    ) -> None:
        """
        Train the DSPO model.
        
        Parameters
        ----------
        dataloader : DataLoader
            Training data loader
        num_steps : int, default=2000
            Number of training steps
        log_every : int, default=100
            Logging frequency
        save_every : int, default=500
            Checkpoint saving frequency
        sample_every : int, default=500
            Sample generation frequency
        output_dir : str, default="outputs/dspo_model"
            Output directory for checkpoints
        sample_prompts : list[str] | None, default=None
            Prompts for sample generation
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare for training
        self.unet, self.optimizer, dataloader = self.accelerator.prepare(
            self.unet, self.optimizer, dataloader
        )
        
        # Default sample prompts
        if sample_prompts is None:
            sample_prompts = [
                "A beautiful sunset over mountains",
                "A futuristic city with flying cars", 
                "A portrait of a wise old wizard",
                "A cute cat playing in a garden",
            ]
        
        logger.info(f"Starting DSPO training for {num_steps} steps")
        
        step = 0
        running_loss = 0.0
        
        with tqdm(total=num_steps, desc="DSPO Training") as pbar:
            while step < num_steps:
                for batch in dataloader:
                    if step >= num_steps:
                        break
                    
                    # Training step
                    metrics = self.train_step(batch)
                    running_loss += metrics["loss"]
                    
                    step += 1
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{metrics['loss']:.4f}")
                    
                    # Logging
                    if step % log_every == 0:
                        avg_loss = running_loss / log_every
                        logger.info(f"Step {step}: loss={avg_loss:.4f}")
                        running_loss = 0.0
                    
                    # Save checkpoint
                    if step % save_every == 0:
                        self.save_checkpoint(output_path / f"checkpoint-{step}")
                    
                    # Generate samples
                    if step % sample_every == 0:
                        self.generate_samples(
                            sample_prompts,
                            output_path / "samples" / f"step_{step}",
                        )
        
        # Save final model
        self.save_checkpoint(output_path / "final")
        logger.info(f"Training completed. Model saved to {output_path}")
    
    def save_checkpoint(self, save_path: Path) -> None:
        """Save model checkpoint."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save DoRA weights
        self.accelerator.unwrap_model(self.unet).save_pretrained(save_path)
        
        # Save optimizer state
        torch.save(
            self.optimizer.state_dict(),
            save_path / "optimizer.pt",
        )
        
        logger.info(f"Checkpoint saved to {save_path}")
    
    def generate_samples(
        self,
        prompts: list[str],
        save_path: Path,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
    ) -> None:
        """Generate sample images for evaluation."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.unet.eval()
        
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                image = self.pipeline(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator().manual_seed(42),
                ).images[0]
                
                image.save(save_path / f"sample_{i:02d}.png")
        
        self.unet.train()
        logger.info(f"Generated {len(prompts)} samples in {save_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        reward_model_path: str,
        **kwargs: Any,
    ) -> "DSPOFineTuner":
        """
        Load fine-tuner from checkpoint.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to model checkpoint
        reward_model_path : str
            Path to reward model
        **kwargs : Any
            Additional arguments for initialization
            
        Returns
        -------
        DSPOFineTuner
            Loaded fine-tuner
        """
        tuner = cls(**kwargs)
        tuner.load_models(reward_model_path)
        
        # Load checkpoint
        checkpoint_path = Path(checkpoint_path)
        tuner.unet = tuner.unet.from_pretrained(checkpoint_path)
        
        logger.info(f"Loaded DSPOFineTuner from {checkpoint_path}")
        return tuner
