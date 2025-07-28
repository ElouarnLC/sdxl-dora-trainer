#!/usr/bin/env python3
"""
Load and use the trained enhanced reward model for DSPO training.
"""

import torch
import logging
from pathlib import Path
from typing import List, Optional
import open_clip

from dspo.enhanced_multimodal_reward import EnhancedMultimodalMultiHeadReward

logger = logging.getLogger(__name__)


class TrainedRewardModel:
    """Wrapper for the trained enhanced reward model."""
    
    def __init__(self, model_path: str):
        """
        Load the trained reward model.
        
        Parameters
        ----------
        model_path : str
            Path to the saved model.pt file
        """
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.config = checkpoint['config']
        self.metrics = checkpoint.get('metrics', {})
        
        logger.info(f"Loading enhanced reward model with F1: {self.metrics.get('f1', 'unknown')}")
        
        # Initialize the model architecture
        self.model = EnhancedMultimodalMultiHeadReward(
            model_name=self.config.get('model_name', 'ViT-L-14-336'),
            fusion_method=self.config.get('fusion_method', 'attention'),
            loss_type=self.config.get('loss_type', 'focal'),
            hidden_dims=self.config.get('hidden_dims', [768, 384, 192]),
            dropout=self.config.get('dropout', 0.05),
            temperature=self.config.get('temperature', 0.1),
        )
        
        # Load the trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Reward model loaded successfully")
        logger.info(f"📊 Architecture: {self.config.get('fusion_method')} fusion, {self.config.get('loss_type')} loss")
        logger.info(f"🏆 Performance: F1={self.metrics.get('f1', 0):.4f}, AUC={self.metrics.get('auc', 0):.4f}")
    
    def get_reward(self, images: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """
        Get reward scores for image-prompt pairs.
        
        Parameters
        ----------
        images : torch.Tensor
            Batch of images [B, C, H, W]
        prompts : List[str]
            List of text prompts
            
        Returns
        -------
        torch.Tensor
            Reward scores [B]
        """
        with torch.no_grad():
            images = images.to(self.device)
            rewards = self.model(images, prompts)
            return rewards.cpu()
    
    def get_individual_head_rewards(self, images: torch.Tensor, prompts: List[str]) -> dict:
        """
        Get individual head rewards for detailed analysis.
        
        Parameters
        ----------
        images : torch.Tensor
            Batch of images [B, C, H, W]
        prompts : List[str]
            List of text prompts
            
        Returns
        -------
        dict
            Dictionary with individual head rewards
        """
        with torch.no_grad():
            images = images.to(self.device)
            head_outputs = self.model(images, prompts, return_individual=True)
            return {k: v.cpu() for k, v in head_outputs.items()}
    
    def compare_images(self, img1: torch.Tensor, img2: torch.Tensor, prompt: str) -> dict:
        """
        Compare two images for the same prompt.
        
        Parameters
        ----------
        img1, img2 : torch.Tensor
            Images to compare [C, H, W]
        prompt : str
            Text prompt
            
        Returns
        -------
        dict
            Comparison results
        """
        images = torch.stack([img1, img2]).unsqueeze(0)  # [1, 2, C, H, W]
        images = images.view(2, *img1.shape)  # [2, C, H, W]
        
        rewards = self.get_reward(images, [prompt, prompt])
        head_rewards = self.get_individual_head_rewards(images, [prompt, prompt])
        
        return {
            'reward_diff': float(rewards[0] - rewards[1]),
            'preference': 'img1' if rewards[0] > rewards[1] else 'img2',
            'confidence': float(torch.abs(rewards[0] - rewards[1])),
            'rewards': rewards.tolist(),
            'head_analysis': {
                head: (float(values[0]), float(values[1]), float(values[0] - values[1]))
                for head, values in head_rewards.items()
            }
        }


def load_reward_model(model_path: str) -> TrainedRewardModel:
    """
    Convenience function to load the trained reward model.
    
    Parameters
    ----------
    model_path : str
        Path to the model.pt file
        
    Returns
    -------
    TrainedRewardModel
        Loaded reward model
    """
    return TrainedRewardModel(model_path)


# Test function
def test_reward_model():
    """Test the loaded reward model."""
    model_path = "outputs/full_enhanced_final/best/model.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    print("🧪 Testing reward model...")
    reward_model = load_reward_model(model_path)
    
    # Create dummy data for testing
    dummy_images = torch.randn(2, 3, 336, 336)  # Match ViT-L-14-336 input size
    dummy_prompts = ["A beautiful landscape", "A cute cat"]
    
    # Test basic reward computation
    rewards = reward_model.get_reward(dummy_images, dummy_prompts)
    print(f"✅ Reward computation works: {rewards}")
    
    # Test individual heads
    head_rewards = reward_model.get_individual_head_rewards(dummy_images, dummy_prompts)
    print(f"✅ Head analysis works: {list(head_rewards.keys())}")
    
    # Test comparison
    comparison = reward_model.compare_images(dummy_images[0], dummy_images[1], "Test prompt")
    print(f"✅ Image comparison works: preference={comparison['preference']}")


if __name__ == "__main__":
    test_reward_model()
