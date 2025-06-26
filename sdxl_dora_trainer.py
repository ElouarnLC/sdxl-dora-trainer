#!/usr/bin/env python3
"""
SDXL DoRA Fine-tuning Tool

A production-ready tool for fine-tuning Stable Diffusion XL models using 
Weight-Decomposed Low-Rank Adaptation (DoRA) technique.

Author: AI Research Community
License: MIT
Version: 1.0.0
"""

import os
import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import wandb
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

# ML Libraries
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from safetensors.torch import save_file, load_file

# Initialize rich console
console = Console()

@dataclass
class TrainingConfig:
    """Configuration class for DoRA training parameters."""
    
    # Model and data paths
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    dataset_path: str = ""
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    
    # DoRA specific parameters
    rank: int = 64
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "to_k", "to_q", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2"
    ])
      # Training parameters
    learning_rate: float = 5e-5  # Reduced from 1e-4 for better stability
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 1000
    warmup_steps: int = 100
    save_every: int = 250
    eval_every: int = 100
    
    # Image parameters
    resolution: int = 1024
    center_crop: bool = True
    random_flip: bool = True
    
    # Optimization
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Scheduler
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    
    # Mixed precision and memory
    mixed_precision: str = "fp16"  # fp16, bf16, no
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = True
    
    # Logging and monitoring
    logging_dir: str = "./logs"
    log_level: str = "INFO"
    report_to: str = "tensorboard"  # tensorboard, wandb, none
    project_name: str = "sdxl-dora-training"
    run_name: Optional[str] = None
    
    # Validation
    validation_prompts: List[str] = field(default_factory=lambda: [
        "a beautiful landscape",
        "a cute cat",
        "abstract art"
    ])
    num_validation_images: int = 4
    validation_steps: int = 100    # Safety and debugging
    enable_safety_checker: bool = True
    debug: bool = False  # Disable debug mode by default
    resume_from_checkpoint: Optional[str] = None
    seed: Optional[int] = None

class CustomImageDataset(Dataset):
    """Custom dataset for loading images and captions."""
    
    def __init__(self, 
                 data_path: str, 
                 tokenizer: CLIPTokenizer,
                 size: int = 1024,
                 center_crop: bool = True,
                 random_flip: bool = True):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.size = size
        
        # Get all image files
        self.image_paths = []
        self.captions = []
        
        # Support multiple formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        
        if self.data_path.is_dir():
            # Directory structure: images and captions
            for img_path in self.data_path.rglob('*'):
                if img_path.suffix.lower() in image_extensions:
                    self.image_paths.append(img_path)
                    
                    # Look for corresponding caption file
                    caption_path = img_path.with_suffix('.txt')
                    if caption_path.exists():
                        with open(caption_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                    else:
                        caption = img_path.stem  # Use filename as caption
                    self.captions.append(caption)
        else:
            raise ValueError(f"Dataset path {data_path} not found or not a directory")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_path}")
        
        # Image transforms
        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])        ])
        
        console.print(f"[green]✓[/green] Loaded {len(self.image_paths)} images from {data_path}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load and process image
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            # Validate image
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError(f"Invalid image size: {image.size}")
            
            image = self.transforms(image)
            
            # Validate transformed image tensor
            if torch.isnan(image).any() or torch.isinf(image).any():
                raise ValueError(f"Invalid values in transformed image from {image_path}")
            
            # Tokenize caption
            caption = self.captions[idx]
            tokens = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": image,
                "input_ids": tokens.input_ids.squeeze(),
                "attention_mask": tokens.attention_mask.squeeze(),
                "caption": caption
            }
        except Exception as e:
            console.print(f"[red]Error loading image {image_path}: {e}[/red]")
            # Return a dummy sample to avoid crashing
            dummy_image = torch.zeros(3, self.size, self.size)
            dummy_tokens = torch.zeros(self.tokenizer.model_max_length, dtype=torch.long)
            return {
                "pixel_values": dummy_image,
                "input_ids": dummy_tokens,
                "attention_mask": dummy_tokens,
                "caption": "dummy caption"
            }

class DoRATrainer:
    """Main trainer class for SDXL DoRA fine-tuning."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.accelerator = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.tokenizer = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.vae = None
        self.noise_scheduler = None
        
        # Training state tracking
        self.consecutive_failures = 0
        self.max_consecutive_failures = 50  # Stop training after 50 consecutive failures
        
        # Setup logging
        self._setup_logging()
        
        # Setup directories
        self._setup_directories()
        
        # Set seed for reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
    
    def _setup_logging(self):
        """Setup logging with rich handler."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_directories(self):
        """Create necessary directories."""
        dirs_to_create = [
            self.config.output_dir,
            self.config.cache_dir,
            self.config.logging_dir,
            os.path.join(self.config.output_dir, "checkpoints"),
            os.path.join(self.config.output_dir, "samples")
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _load_models(self):
        """Load and setup SDXL models."""
        console.print(Panel("[bold blue]Loading SDXL Models[/bold blue]"))
        
        try:
            # Load tokenizer and text encoders (SDXL has two)
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.config.model_name,
                subfolder="tokenizer",
                cache_dir=self.config.cache_dir
            )
            
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                self.config.model_name,
                subfolder="tokenizer_2",
                cache_dir=self.config.cache_dir
            )
            
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.config.model_name,
                subfolder="text_encoder",
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float16 if self.config.mixed_precision == "fp16" else torch.float32
            )
            
            self.text_encoder_2 = CLIPTextModel.from_pretrained(
                self.config.model_name,
                subfolder="text_encoder_2",
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float16 if self.config.mixed_precision == "fp16" else torch.float32
            )
              # Load VAE (always use float32 to avoid NaN issues)
            self.vae = AutoencoderKL.from_pretrained(
                self.config.model_name,
                subfolder="vae",
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float32  # Always use float32 for VAE to prevent NaN
            )
            
            # Load UNet
            self.model = UNet2DConditionModel.from_pretrained(
                self.config.model_name,
                subfolder="unet",
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float16 if self.config.mixed_precision == "fp16" else torch.float32
            )
            
            # Setup noise scheduler
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                self.config.model_name,
                subfolder="scheduler"
            )            # Freeze models except UNet
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)
            
            # Set models to eval mode (except UNet which will be trained)
            self.vae.eval()
            self.text_encoder.eval()
            self.text_encoder_2.eval()
            
            # Enable gradient checkpointing
            if self.config.gradient_checkpointing:
                self.model.enable_gradient_checkpointing()
                if hasattr(self.text_encoder, "gradient_checkpointing_enable"):
                    self.text_encoder.gradient_checkpointing_enable()
                if hasattr(self.text_encoder_2, "gradient_checkpointing_enable"):
                    self.text_encoder_2.gradient_checkpointing_enable()
            
            console.print("[green]✓[/green] Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
    
    def _setup_dora(self):
        """Setup DoRA (Weight-Decomposed Low-Rank Adaptation)."""
        console.print(Panel("[bold blue]Setting up DoRA[/bold blue]"))
        
        try:            # DoRA configuration
            dora_config = LoraConfig(
                r=self.config.rank,
                lora_alpha=self.config.alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.dropout,
                bias="none",
                use_dora=True  # This enables DoRA instead of LoRA
            )
            
            # Apply DoRA to the model
            self.model = get_peft_model(self.model, dora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            console.print("[green]✓[/green] DoRA setup complete")
            console.print(f"Trainable parameters: {trainable_params:,}")
            console.print(f"Total parameters: {total_params:,}")
            console.print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
            
        except Exception as e:
            self.logger.error(f"Failed to setup DoRA: {e}")
            raise
    
    def _create_dataset(self):
        """Create training dataset."""
        console.print(Panel("[bold blue]Creating Dataset[/bold blue]"))
        
        try:
            dataset = CustomImageDataset(
                data_path=self.config.dataset_path,
                tokenizer=self.tokenizer,
                size=self.config.resolution,
                center_crop=self.config.center_crop,
                random_flip=self.config.random_flip
            )
            
            self.train_dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            console.print(f"[green]✓[/green] Dataset created with {len(dataset)} samples")
            
        except Exception as e:
            self.logger.error(f"Failed to create dataset: {e}")
            raise
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        console.print(Panel("[bold blue]Setting up Optimizer[/bold blue]"))
        
        try:
            # Get trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            
            # Setup optimizer
            if self.config.optimizer.lower() == "adamw":
                if self.config.use_8bit_adam:
                    try:
                        import bitsandbytes as bnb
                        self.optimizer = bnb.optim.AdamW8bit(
                            trainable_params,
                            lr=self.config.learning_rate,
                            betas=(self.config.beta1, self.config.beta2),
                            weight_decay=self.config.weight_decay,
                            eps=self.config.epsilon
                        )
                    except ImportError:
                        self.logger.warning("bitsandbytes not available, using regular AdamW")
                        self.optimizer = torch.optim.AdamW(
                            trainable_params,
                            lr=self.config.learning_rate,
                            betas=(self.config.beta1, self.config.beta2),
                            weight_decay=self.config.weight_decay,
                            eps=self.config.epsilon
                        )
                else:
                    self.optimizer = torch.optim.AdamW(
                        trainable_params,
                        lr=self.config.learning_rate,
                        betas=(self.config.beta1, self.config.beta2),
                        weight_decay=self.config.weight_decay,
                        eps=self.config.epsilon
                    )
            else:
                raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
            
            # Setup learning rate scheduler
            if self.config.lr_scheduler.lower() == "cosine":
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.max_train_steps
                )
            elif self.config.lr_scheduler.lower() == "linear":
                self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    total_iters=self.config.lr_warmup_steps
                )
            
            console.print("[green]✓[/green] Optimizer and scheduler setup complete")
            
        except Exception as e:
            self.logger.error(f"Failed to setup optimizer: {e}")
            raise
    
    def _setup_accelerator(self):
        """Setup Accelerate for distributed training."""
        console.print(Panel("[bold blue]Setting up Accelerator[/bold blue]"))
        
        try:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                mixed_precision=self.config.mixed_precision,
                log_with=self.config.report_to if self.config.report_to != "none" else None,
                project_dir=self.config.logging_dir
            )
            
            # Prepare models and optimizers
            (
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler
            ) = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler
            )
              # Move other models to device
            self.vae = self.vae.to(self.accelerator.device)
            self.text_encoder = self.text_encoder.to(self.accelerator.device)
            self.text_encoder_2 = self.text_encoder_2.to(self.accelerator.device)
            
            console.print(f"[green]✓[/green] Accelerator setup complete (Device: {self.accelerator.device})")
            
        except Exception as e:
            self.logger.error(f"Failed to setup accelerator: {e}")
            raise
    
    def _setup_logging_tools(self):
        """Setup logging tools (wandb, tensorboard)."""
        if self.config.report_to == "wandb":
            try:
                self.accelerator.init_trackers(
                    project_name=self.config.project_name,
                    config=self.config.__dict__,
                    init_kwargs={"wandb": {"name": self.config.run_name}}
                )
                console.print("[green]✓[/green] Weights & Biases logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to setup wandb: {e}")
        
        elif self.config.report_to == "tensorboard":
            try:
                # Filter config to only include serializable values for tensorboard
                config_dict = {}
                for key, value in self.config.__dict__.items():
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        config_dict[key] = value
                    elif isinstance(value, list) and all(isinstance(x, (int, float, str, bool)) for x in value):
                        config_dict[key] = value
                    else:
                        config_dict[key] = str(value)
                
                self.accelerator.init_trackers(
                    project_name=self.config.project_name,
                    config=config_dict
                )
                console.print("[green]✓[/green] TensorBoard logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to setup tensorboard: {e}")
    
    def compute_loss(self, batch):
        """Compute the diffusion loss."""
        try:
            # Get the target dtype for consistency
            target_dtype = torch.float16 if self.config.mixed_precision == "fp16" else torch.float32
            
            # Ensure input images are in the correct dtype and device
            pixel_values = batch["pixel_values"].to(self.accelerator.device, dtype=target_dtype)            # Encode images to latent space
            with torch.no_grad():
                # Ensure pixel values are in valid range for VAE
                pixel_values = torch.clamp(pixel_values, -1.0, 1.0)                # Debug: Check input pixel values
                if self.config.debug:
                    self.logger.debug(f"pixel_values stats: min={pixel_values.min():.3f}, max={pixel_values.max():.3f}, shape={pixel_values.shape}")
                    self.logger.debug(f"pixel_values dtype: {pixel_values.dtype}, device: {pixel_values.device}")
                    self.logger.debug(f"VAE dtype: {next(self.vae.parameters()).dtype}, device: {next(self.vae.parameters()).device}")
                
                # Try VAE encoding with error handling
                try:
                    # Ensure pixel values are in float32 for VAE to avoid precision issues
                    pixel_values_for_vae = pixel_values.float()
                    latents = self.vae.encode(pixel_values_for_vae).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    
                    # Convert back to target dtype
                    latents = latents.to(target_dtype)
                    
                except Exception as e:
                    self.logger.error(f"VAE encoding failed: {e}")
                    return None
                  # Check for NaN in latents
                if torch.isnan(latents).any():
                    self.consecutive_failures += 1
                    self.logger.error(f"NaN detected in VAE latents (failure #{self.consecutive_failures})")
                    
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        self.logger.error(f"Training stopped: {self.consecutive_failures} consecutive failures detected")
                        self.logger.error("This indicates a systematic issue with your dataset or VAE processing")
                        self.logger.error("Please check your images and ensure they are valid")
                        raise RuntimeError(f"Too many consecutive failures ({self.consecutive_failures})")
                    
                    if self.config.debug:
                        self.logger.debug(f"Original latents shape: {latents.shape}")
                        self.logger.debug(f"NaN count: {torch.isnan(latents).sum().item()}")
                        self.logger.debug(f"Inf count: {torch.isinf(latents).sum().item()}")
                    return None  # Skip this batch
                
                # Reset failure counter on success
                self.consecutive_failures = 0
                
                # Debug: Check latent values
                if self.config.debug:
                    self.logger.debug(f"latents stats: min={latents.min():.3f}, max={latents.max():.3f}, shape={latents.shape}")
            
            # Add noise to latents
            noise = torch.randn_like(latents, dtype=target_dtype, device=latents.device)
            # Clamp noise to prevent extreme values
            noise = torch.clamp(noise, -3.0, 3.0)
            
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=latents.device
            ).long()
            
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)# Encode text with both encoders
            with torch.no_grad():
                input_ids = batch["input_ids"].to(self.accelerator.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.accelerator.device)
                
                # For SDXL, we use a different approach
                # We need to encode text properly for SDXL
                
                # First text encoder (CLIP ViT-L/14)
                encoder_output_1 = self.text_encoder(
                    input_ids, 
                    attention_mask=attention_mask
                )
                prompt_embeds_1 = encoder_output_1.last_hidden_state
                
                # Second text encoder (OpenCLIP ViT-bigG/14)
                encoder_output_2 = self.text_encoder_2(
                    input_ids,
                    attention_mask=attention_mask
                )
                prompt_embeds_2 = encoder_output_2.last_hidden_state
                pooled_prompt_embeds = encoder_output_2.pooler_output
                  # For SDXL, we concatenate the text embeddings along the feature dimension
                # Make sure both have the same sequence length
                seq_len = min(prompt_embeds_1.shape[1], prompt_embeds_2.shape[1])
                prompt_embeds_1 = prompt_embeds_1[:, :seq_len, :]
                prompt_embeds_2 = prompt_embeds_2[:, :seq_len, :]
                
                # Debug: Print embedding shapes
                if self.config.debug:
                    self.logger.debug(f"prompt_embeds_1 shape: {prompt_embeds_1.shape}")
                    self.logger.debug(f"prompt_embeds_2 shape: {prompt_embeds_2.shape}")
                    self.logger.debug(f"pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")
                
                # Concatenate along the feature dimension (768 + 1280 = 2048)
                encoder_hidden_states = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
              # For SDXL, we need to provide additional conditioning
            batch_size = encoder_hidden_states.shape[0]
            
            # Debug: Print tensor shapes
            if self.config.debug:
                self.logger.debug(f"encoder_hidden_states shape: {encoder_hidden_states.shape}")
                self.logger.debug(f"noisy_latents shape: {noisy_latents.shape}")
                self.logger.debug(f"timesteps shape: {timesteps.shape}")
            
            # Create time embeddings for SDXL
            # SDXL expects original size, crop coordinates, and target size
            original_size = torch.tensor([[self.config.resolution, self.config.resolution]] * batch_size).to(self.accelerator.device)
            crops_coords_top_left = torch.tensor([[0, 0]] * batch_size).to(self.accelerator.device)
            target_size = torch.tensor([[self.config.resolution, self.config.resolution]] * batch_size).to(self.accelerator.device)
            
            # Create added conditioning kwargs for SDXL
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds.to(target_dtype),
                "time_ids": torch.cat([original_size, crops_coords_top_left, target_size], dim=1).to(target_dtype)
            }
              # Predict noise
            model_pred = self.model(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs
            ).sample
            
            # Debug: Check for NaN values
            if self.config.debug:
                self.logger.debug(f"model_pred stats: min={model_pred.min():.4f}, max={model_pred.max():.4f}, has_nan={torch.isnan(model_pred).any()}")
                self.logger.debug(f"noise stats: min={noise.min():.4f}, max={noise.max():.4f}, has_nan={torch.isnan(noise).any()}")
            
            # Check for NaN values and handle them
            if torch.isnan(model_pred).any() or torch.isnan(noise).any():
                self.logger.warning("NaN detected in model prediction or noise - skipping this batch")
                return torch.tensor(0.0, device=model_pred.device, requires_grad=True)
            
            # Compute loss (ensure both tensors have the same dtype and are finite)
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
              # Check if loss is finite
            if not torch.isfinite(loss):
                self.logger.warning(f"Non-finite loss detected: {loss}, replacing with zero")
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
            
            return loss
            
        except Exception as e:
            self.logger.error(f"Error computing loss: {e}")
            raise
    
    def validate(self, step: int):
        """Run validation and generate sample images."""
        try:
            self.model.eval()
            
            # Get the base UNet from the PEFT model
            base_unet = self.accelerator.unwrap_model(self.model)
            if hasattr(base_unet, 'get_base_model'):
                base_unet = base_unet.get_base_model()
            
            # Create validation pipeline
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.config.model_name,
                unet=base_unet,
                torch_dtype=torch.float16 if self.config.mixed_precision == "fp16" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipeline = pipeline.to(self.accelerator.device)
            
            # Generate validation images
            validation_images = []
            for prompt in self.config.validation_prompts:
                with torch.autocast("cuda"):
                    image = pipeline(
                        prompt,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        num_images_per_prompt=1
                    ).images[0]
                    validation_images.append((prompt, image))
            
            # Save validation images
            validation_dir = Path(self.config.output_dir) / "samples" / f"step_{step}"
            validation_dir.mkdir(parents=True, exist_ok=True)
            
            for i, (prompt, image) in enumerate(validation_images):
                image.save(validation_dir / f"sample_{i}_{prompt[:50].replace(' ', '_')}.png")
            
            # Log to tracking service
            if self.config.report_to != "none":
                log_dict = {
                    "validation_images": [
                        wandb.Image(image, caption=prompt) 
                        for prompt, image in validation_images
                    ]
                }
                self.accelerator.log(log_dict, step=step)
            
            console.print(f"[green]✓[/green] Validation complete at step {step}")
            
            # Cleanup
            del pipeline
            torch.cuda.empty_cache()
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
        finally:
            self.model.train()
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        try:
            checkpoint_dir = Path(self.config.output_dir) / "checkpoints" / f"checkpoint-{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save DoRA weights
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(checkpoint_dir)
            
            # Save training state
            self.accelerator.save_state(checkpoint_dir / "accelerator_state")
            
            # Save config
            with open(checkpoint_dir / "training_config.json", "w") as f:
                json.dump(self.config.__dict__, f, indent=2, default=str)
            
            console.print(f"[green]✓[/green] Checkpoint saved at step {step}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def train(self):
        """Main training loop."""
        console.print(Panel("[bold green]Starting DoRA Training[/bold green]"))
        
        try:            # Setup everything
            self._load_models()
            self._setup_dora()
            self._create_dataset()
            self._setup_optimizer()
            self._setup_accelerator()
            self._setup_logging_tools()
            
            # Test VAE encoding to catch issues early
            if not self.test_vae_encoding():
                raise RuntimeError("VAE encoding test failed - training cannot proceed")
            
            # Training metrics
            global_step = 0
            total_loss = 0.0
            
            # Progress tracking
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            )
            
            with progress:
                task = progress.add_task("Training...", total=self.config.max_train_steps)
                  # Training loop
                for epoch in range(1000):  # Large number, will break when max_steps reached
                    for batch in self.train_dataloader:
                        with self.accelerator.accumulate(self.model):
                            # Compute loss
                            loss = self.compute_loss(batch)
                            
                            # Skip batch if loss computation failed
                            if loss is None:
                                self.logger.warning("Skipping batch due to data issues")
                                continue
                            
                            # Backward pass
                            self.accelerator.backward(loss)
                            
                            # Gradient clipping
                            if self.accelerator.sync_gradients:
                                self.accelerator.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.config.max_grad_norm
                                )
                            
                            # Optimizer step
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad()
                        
                        # Update metrics only if we have a valid loss
                        if loss is not None and torch.isfinite(loss):
                            total_loss += loss.detach().item()
                        global_step += 1
                        
                        # Logging
                        if global_step % 10 == 0:
                            avg_loss = total_loss / min(global_step, 10) if total_loss > 0 else 0.0
                            lr = self.lr_scheduler.get_last_lr()[0]
                            
                            progress.update(
                                task,
                                advance=1,
                                description=f"Training (Loss: {avg_loss:.4f}, LR: {lr:.2e})"
                            )
                            
                            if self.config.report_to != "none":
                                log_dict = {
                                    "train_loss": avg_loss,
                                    "learning_rate": lr,
                                    "step": global_step
                                }
                                self.accelerator.log(log_dict, step=global_step)
                        
                        # Validation
                        if global_step % self.config.validation_steps == 0:
                            self.validate(global_step)
                        
                        # Save checkpoint
                        if global_step % self.config.save_every == 0:
                            self.save_checkpoint(global_step)
                        
                        # Check if training is complete
                        if global_step >= self.config.max_train_steps:
                            break
                    
                    if global_step >= self.config.max_train_steps:
                        break
            
            # Final checkpoint and validation
            self.save_checkpoint(global_step)
            self.validate(global_step)
            
            console.print(Panel("[bold green]Training Complete![/bold green]"))
            console.print(f"Final model saved to: {self.config.output_dir}")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Training interrupted by user[/yellow]")
            self.save_checkpoint(global_step)
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            traceback.print_exc()
            raise
        finally:
            # Cleanup
            if self.accelerator:
                self.accelerator.end_training()

    def test_vae_encoding(self):
        """Test VAE encoding with sample data to diagnose issues."""
        try:
            console.print("[yellow]Testing VAE encoding...[/yellow]")
            
            # Create test data
            test_batch = torch.randn(1, 3, 1024, 1024, device=self.accelerator.device, dtype=torch.float32)
            test_batch = torch.clamp(test_batch, -1.0, 1.0)
            
            with torch.no_grad():
                # Test VAE encoding
                latents = self.vae.encode(test_batch).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                
                # Check results
                if torch.isnan(latents).any():
                    self.logger.error("VAE test failed: NaN detected with synthetic data")
                    return False
                
                console.print(f"[green]✓[/green] VAE test passed. Latents shape: {latents.shape}, range: [{latents.min():.3f}, {latents.max():.3f}]")
                return True
                
        except Exception as e:
            self.logger.error(f"VAE test failed: {e}")
            return False

def create_config_from_args(args) -> TrainingConfig:
    """Create training configuration from command line arguments."""
    config = TrainingConfig()
    
    # Update config with provided arguments
    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    
    return config

def validate_config(config: TrainingConfig) -> List[str]:
    """Validate training configuration and return list of errors."""
    errors = []
    
    # Check required paths
    if not config.dataset_path:
        errors.append("Dataset path is required")
    elif not Path(config.dataset_path).exists():
        errors.append(f"Dataset path does not exist: {config.dataset_path}")
    
    # Check model name
    if not config.model_name:
        errors.append("Model name is required")
    
    # Check numerical parameters
    if config.rank <= 0:
        errors.append("Rank must be positive")
    
    if config.alpha <= 0:
        errors.append("Alpha must be positive")
    
    if config.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    if config.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if config.max_train_steps <= 0:
        errors.append("Max train steps must be positive")
    
    # Check resolution
    if config.resolution not in [512, 768, 1024]:
        errors.append("Resolution must be 512, 768, or 1024")
    
    # Check mixed precision
    if config.mixed_precision not in ["fp16", "bf16", "no"]:
        errors.append("Mixed precision must be fp16, bf16, or no")
    
    return errors

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SDXL DoRA Fine-tuning Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,        epilog="""
Examples:
  # Using configuration file
  python sdxl_dora_trainer.py --config config.yaml
  
  # Basic training with command line arguments
  python sdxl_dora_trainer.py --dataset_path ./my_images --output_dir ./output
  
  # Override config file with command line arguments
  python sdxl_dora_trainer.py --config config.yaml --learning_rate 5e-5 --batch_size 2
  
  # Advanced training with custom parameters
  python sdxl_dora_trainer.py \\
    --dataset_path ./my_images \\
    --output_dir ./output \\
    --rank 128 \\
    --alpha 64 \\
    --learning_rate 5e-5 \\
    --batch_size 2 \\
    --max_train_steps 2000 \\
    --resolution 1024 \\
    --mixed_precision fp16 \\
    --report_to wandb \\
    --project_name my-sdxl-project
        """
    )
      # Required arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the training dataset directory"
    )
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML or JSON)"
    )
    
    # Model and output
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for checkpoints and samples")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                       help="Cache directory for downloaded models")
    
    # DoRA parameters
    parser.add_argument("--rank", type=int, default=64,
                       help="DoRA rank (default: 64)")
    parser.add_argument("--alpha", type=int, default=32,
                       help="DoRA alpha parameter (default: 32)")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="DoRA dropout rate (default: 0.1)")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size (default: 1)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--max_train_steps", type=int, default=1000,
                       help="Maximum training steps (default: 1000)")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Learning rate warmup steps (default: 100)")
    
    # Image parameters
    parser.add_argument("--resolution", type=int, default=1024, choices=[512, 768, 1024],
                       help="Training image resolution (default: 1024)")
    parser.add_argument("--center_crop", action="store_true", default=True,
                       help="Center crop images")
    parser.add_argument("--random_flip", action="store_true", default=True,
                       help="Random horizontal flip")
    
    # Optimization
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw"],
                       help="Optimizer to use (default: adamw)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay (default: 0.01)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm (default: 1.0)")
    
    # Mixed precision and memory
    parser.add_argument("--mixed_precision", type=str, default="fp16", 
                       choices=["fp16", "bf16", "no"],
                       help="Mixed precision training (default: fp16)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                       help="Enable gradient checkpointing")
    parser.add_argument("--use_8bit_adam", action="store_true", default=True,
                       help="Use 8-bit Adam optimizer")
    
    # Logging and monitoring
    parser.add_argument("--logging_dir", type=str, default="./logs",
                       help="Logging directory (default: ./logs)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                       choices=["tensorboard", "wandb", "none"],
                       help="Logging service (default: tensorboard)")
    parser.add_argument("--project_name", type=str, default="sdxl-dora-training",
                       help="Project name for logging")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Run name for logging")
    
    # Validation
    parser.add_argument("--validation_steps", type=int, default=100,
                       help="Steps between validations (default: 100)")
    parser.add_argument("--save_every", type=int, default=250,
                       help="Steps between checkpoints (default: 250)")
      # Safety and debugging
    parser.add_argument("--enable_safety_checker", action="store_true", default=True,
                       help="Enable safety checker")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Resume from checkpoint path")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
      # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # Load from config file
        try:
            # Import here to avoid circular imports
            from config_manager import ConfigManager
            config = ConfigManager.load_config(args.config)
            
            # Override with command line arguments
            for key, value in vars(args).items():
                if key != 'config' and value is not None:
                    setattr(config, key, value)
        except Exception as e:
            console.print(f"[red]Failed to load config file: {e}[/red]")
            sys.exit(1)
    else:
        # Create config from command line arguments
        if not args.dataset_path:
            console.print("[red]Error: --dataset_path or --config is required[/red]")
            parser.print_help()
            sys.exit(1)
        config = create_config_from_args(args)    # Validate configuration
    errors = validate_config(config)
    
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  • {error}")
        sys.exit(1)
    
    # Print configuration
    table = Table(title="Training Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    key_params = [
        ("Dataset Path", config.dataset_path),
        ("Model Name", config.model_name),
        ("Output Directory", config.output_dir),
        ("DoRA Rank", config.rank),
        ("DoRA Alpha", config.alpha),
        ("Learning Rate", config.learning_rate),
        ("Batch Size", config.batch_size),
        ("Max Steps", config.max_train_steps),
        ("Resolution", config.resolution),
        ("Mixed Precision", config.mixed_precision),
        ("Report To", config.report_to)
    ]
    
    for param, value in key_params:
        table.add_row(param, str(value))
    
    console.print(table)
      # Confirm training start
    if not config.debug:
        response = input("\nStart training? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            console.print("Training cancelled.")
            sys.exit(0)
      # Create and run trainer
    try:
        trainer = DoRATrainer(config)
        trainer.train()
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        if config.debug:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
