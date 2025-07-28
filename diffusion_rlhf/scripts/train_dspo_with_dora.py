#!/usr/bin/env python3
"""
Enhanced DSPO training with DoRA fine-tuning and trained reward model.

This script performs Direct Statistical Preference Optimization (DSPO) training
using DoRA (Weight-Decomposed Low-Rank Adaptation) for SDXL fine-tuning
with your high-performance reward model (F1=0.8333).

Uses the existing DSPOFineTuner class which properly implements DSPO training.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Import existing DSPO infrastructure
from dspo.tuner import DSPOFineTuner


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PreferencePairDataset(Dataset):
    """Dataset for DSPO preference pairs."""
    
    def __init__(self, pairs: list[dict[str, Any]]):
        """
        Initialize preference pair dataset.
        
        Parameters
        ----------
        pairs : list[dict[str, Any]]
            List of preference pairs with keys:
            - prompt: text prompt
            - image_a: preferred image 
            - image_b: dispreferred image
            - preference: which image is preferred
        """
        self.pairs = pairs
        
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair = self.pairs[idx]
        
        # Determine which image is preferred
        if pair['preference'] == 'img1':
            img_pos = self._pil_to_tensor(pair['image_a'])
            img_neg = self._pil_to_tensor(pair['image_b'])
        else:
            img_pos = self._pil_to_tensor(pair['image_b'])
            img_neg = self._pil_to_tensor(pair['image_a'])
            
        return {
            'prompt': pair['prompt'],
            'img_pos': img_pos,
            'img_neg': img_neg,
        }
    
    def _pil_to_tensor(self, image) -> torch.Tensor:
        """Convert PIL image to tensor."""
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        return transform(image)


def generate_preference_pairs(
    model_name: str,
    reward_model_path: str,
    prompts: list[str],
    num_pairs: int = 100,
    device: str = "auto"
) -> list[dict[str, Any]]:
    """
    Generate preference pairs using different generation parameters.
    
    Parameters
    ----------
    model_name : str
        SDXL model name
    reward_model_path : str  
        Path to reward model
    prompts : list[str]
        List of prompts
    num_pairs : int
        Number of pairs to generate
    device : str
        Device to use
        
    Returns
    -------
    list[dict[str, Any]]
        Generated preference pairs
    """
    from diffusers import StableDiffusionXLPipeline
    from dspo.trained_reward_model import load_reward_model
    
    logger.info(f"🎨 Generating {num_pairs} preference pairs...")
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load pipeline for generation
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
    )
    pipe = pipe.to(device)
    
    # Load reward model
    reward_model = load_reward_model(reward_model_path)
    
    pairs = []
    guidance_scales = [7.5, 10.0, 12.0]
    inference_steps = [20, 30, 50]
    
    for i in range(num_pairs):
        # Sample prompt and parameters
        prompt = np.random.choice(prompts)
        cfg1, cfg2 = np.random.choice(guidance_scales, size=2, replace=True)
        steps1, steps2 = np.random.choice(inference_steps, size=2, replace=True)
        
        # Generate two images
        with torch.no_grad():
            img_a = pipe(
                prompt=prompt,
                guidance_scale=cfg1,
                num_inference_steps=steps1,
                height=1024,
                width=1024,
                generator=torch.Generator().manual_seed(i),
            ).images[0]
            
            img_b = pipe(
                prompt=prompt,
                guidance_scale=cfg2,
                num_inference_steps=steps2,
                height=1024,
                width=1024,
                generator=torch.Generator().manual_seed(i + 1000),
            ).images[0]
        
        # Get preference using reward model
        comparison = reward_model.compare_images(
            _pil_to_reward_tensor(img_a),
            _pil_to_reward_tensor(img_b),
            prompt
        )
        
        pairs.append({
            'prompt': prompt,
            'image_a': img_a,
            'image_b': img_b,
            'preference': comparison['preference'],
            'reward_diff': comparison['reward_diff'],
            'confidence': comparison['confidence'],
        })
        
        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i+1}/{num_pairs} pairs")
    
    logger.info(f"✅ Generated {len(pairs)} preference pairs")
    return pairs


def _pil_to_reward_tensor(image):
    """Convert PIL image to tensor for reward model."""
    import torchvision.transforms as transforms
    
    # Resize to reward model input size (336x336 for ViT-L-14-336)
    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    return transform(image)


def main():
    """Main DSPO training function using existing DSPOFineTuner."""
    parser = argparse.ArgumentParser(description="Enhanced DSPO training with DoRA")
    parser.add_argument("--prompts", type=str, required=True, 
                       help="Path to prompts CSV file")
    parser.add_argument("--output", type=str, default="outputs/dspo_dora", 
                       help="Output directory")
    parser.add_argument("--reward-model", type=str, 
                       default="outputs/full_enhanced_final/best/model.pt",
                       help="Path to trained reward model")
    parser.add_argument("--model-name", type=str,
                       default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="SDXL model name")
    parser.add_argument("--num-pairs", type=int, default=200, 
                       help="Number of preference pairs")
    parser.add_argument("--num-steps", type=int, default=2000,
                       help="Number of training steps")
    parser.add_argument("--lr", type=float, default=5e-5, 
                       help="Learning rate")
    parser.add_argument("--rank", type=int, default=32,
                       help="DoRA rank")
    parser.add_argument("--alpha", type=int, default=None,
                       help="DoRA alpha (defaults to 4 * rank)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--beta", type=float, default=0.1,
                       help="DSPO beta parameter")
    
    args = parser.parse_args()
    
    logger.info("🚀 Enhanced DSPO Training with DoRA")
    logger.info("=" * 50)
    logger.info(f"📱 Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"🤖 Model: {args.model_name}")
    logger.info(f"🏆 Reward Model: {args.reward_model}")
    logger.info(f"📊 DoRA: rank={args.rank}, alpha={args.alpha}")
    
    # Load prompts
    if args.prompts.endswith('.csv'):
        prompts_df = pd.read_csv(args.prompts)
        if 'prompt' in prompts_df.columns:
            prompts = prompts_df['prompt'].tolist()
        else:
            prompts = prompts_df.iloc[:, 0].tolist()
    else:
        with open(args.prompts) as f:
            prompts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"📝 Loaded {len(prompts)} prompts")
    
    # Generate preference pairs
    pairs = generate_preference_pairs(
        model_name=args.model_name,
        reward_model_path=args.reward_model,
        prompts=prompts,
        num_pairs=args.num_pairs,
    )
    
    # Create dataset and dataloader
    dataset = PreferencePairDataset(pairs)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues with PIL
    )
    
    # Initialize DSPO fine-tuner with DoRA
    tuner = DSPOFineTuner(
        model_name=args.model_name,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.lr,
        beta=args.beta,
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        gradient_checkpointing=True,
    )
    
    # Load models
    tuner.load_models(
        reward_model_path=args.reward_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Setup optimizer
    tuner.setup_optimizer()
    
    # Sample prompts for evaluation
    sample_prompts = prompts[:4] if len(prompts) >= 4 else prompts
    
    # Start training
    logger.info(f"🚀 Starting DSPO training for {args.num_steps} steps...")
    
    tuner.train(
        dataloader=dataloader,
        num_steps=args.num_steps,
        log_every=100,
        save_every=500,
        sample_every=500,
        output_dir=args.output,
        sample_prompts=sample_prompts,
    )
    
    logger.info(f"✅ DSPO training completed! Model saved to {args.output}")
    logger.info("\n🎯 Next steps:")
    logger.info(f"  • Load fine-tuned model from {args.output}/final")
    logger.info("  • Use DoRA weights for improved image generation")
    logger.info("  • Compare results with base SDXL model")


if __name__ == "__main__":
    main()
