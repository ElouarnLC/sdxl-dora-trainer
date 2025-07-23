"""
Multi-head reward model for image preference learning.

This module implements a reward model with frozen OpenCLIP ViT-L/14 backbone
and 5 specialized MLP heads for different aspects of image quality.
"""

import logging
from pathlib import Path
from typing import Any

import open_clip
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


class MLPHead(nn.Module):
    """
    Multi-layer perceptron head for reward prediction.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dim : int, default=512
        Hidden layer dimension
    output_dim : int, default=1
        Output dimension (1 for scalar reward)
    dropout : float, default=0.1
        Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP head."""
        return self.layers(x)


class MultiHeadReward(nn.Module):
    """
    Multi-head reward model with frozen OpenCLIP backbone.
    
    Uses ViT-L/14 as frozen feature extractor with 5 specialized heads:
    - spatial: Spatial composition and layout
    - icono: Iconographic elements and symbols  
    - style: Artistic style and aesthetics
    - fidelity: Image quality and fidelity
    - material: Material properties and textures
    
    Parameters
    ----------
    model_name : str, default="ViT-L-14"
        OpenCLIP model name
    pretrained : str, default="laion2b_s32b_b82k"
        Pretrained weights dataset
    hidden_dim : int, default=512
        Hidden dimension for MLP heads
    dropout : float, default=0.1
        Dropout probability
    freeze_backbone : bool, default=True
        Whether to freeze the backbone
    """
    
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        
        # Load OpenCLIP model
        self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        # Get feature dimension
        feature_dim = self.backbone.visual.output_dim
        
        # Create specialized heads
        self.heads = nn.ModuleDict({
            "spatial": MLPHead(feature_dim, hidden_dim, 1, dropout),
            "icono": MLPHead(feature_dim, hidden_dim, 1, dropout),
            "style": MLPHead(feature_dim, hidden_dim, 1, dropout),
            "fidelity": MLPHead(feature_dim, hidden_dim, 1, dropout),
            "material": MLPHead(feature_dim, hidden_dim, 1, dropout),
        })
        
        # Learnable weights for combining heads
        self.head_weights = nn.Parameter(torch.ones(5) / 5)
        
        logger.info(f"Initialized MultiHeadReward with {model_name} backbone")
        logger.info(f"Feature dimension: {feature_dim}")
        logger.info(f"Backbone frozen: {freeze_backbone}")
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images using frozen backbone.
        
        Parameters
        ----------
        images : torch.Tensor
            Batch of images [B, C, H, W]
            
        Returns
        -------
        torch.Tensor
            Image features [B, feature_dim]
        """
        with torch.set_grad_enabled(self.training and not self._backbone_frozen()):
            features = self.backbone.encode_image(images)
            
        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def _backbone_frozen(self) -> bool:
        """Check if backbone is frozen."""
        return not next(self.backbone.parameters()).requires_grad
    
    def forward(
        self, 
        images: torch.Tensor,
        return_individual: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass through reward model.
        
        Parameters
        ----------
        images : torch.Tensor
            Batch of images [B, C, H, W]
        return_individual : bool, default=False
            Whether to return individual head outputs
            
        Returns
        -------
        torch.Tensor or dict
            If return_individual=False: Combined reward scores [B, 1]
            If return_individual=True: Dict with individual head outputs
        """
        # Extract features
        features = self.extract_features(images)
        
        # Get individual head outputs
        head_outputs = {}
        for name, head in self.heads.items():
            head_outputs[name] = head(features)
        
        if return_individual:
            return head_outputs
        
        # Combine outputs using learnable weights
        weighted_outputs = []
        weights = torch.softmax(self.head_weights, dim=0)
        
        for i, (name, output) in enumerate(head_outputs.items()):
            weighted_outputs.append(weights[i] * output)
        
        combined_reward = torch.stack(weighted_outputs, dim=-1).sum(dim=-1)
        return combined_reward
    
    def compute_preference_loss(
        self,
        img_pos: torch.Tensor,
        img_neg: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute preference loss for training.
        
        Parameters
        ----------
        img_pos : torch.Tensor
            Positive images [B, C, H, W]
        img_neg : torch.Tensor
            Negative images [B, C, H, W]
        labels : torch.Tensor
            Preference labels [B] or [B, 5] for multi-head
            
        Returns
        -------
        torch.Tensor
            Preference loss scalar
        """
        # Get reward scores
        reward_pos = self.forward(img_pos)
        reward_neg = self.forward(img_neg)
        
        # Compute preference logits
        logits = reward_pos - reward_neg
        
        # Binary cross-entropy loss
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(), labels
        )
        
        return loss
    
    def compute_multihead_preference_loss(
        self,
        img_pos: torch.Tensor,
        img_neg: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute multi-head preference loss for training.
        
        Parameters
        ----------
        img_pos : torch.Tensor
            Positive images [B, C, H, W]
        img_neg : torch.Tensor
            Negative images [B, C, H, W]
        labels : torch.Tensor
            Multi-head preference labels [B, 5]
            
        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with individual head losses and total loss
        """
        # Get individual head outputs
        head_outputs_pos = self.forward(img_pos, return_individual=True)
        head_outputs_neg = self.forward(img_neg, return_individual=True)
        
        # Compute loss for each head
        losses = {}
        total_loss = 0
        head_names = ["spatial", "icono", "style", "fidelity", "material"]
        
        for i, head_name in enumerate(head_names):
            # Get head-specific logits
            logits = head_outputs_pos[head_name] - head_outputs_neg[head_name]
            
            # Get head-specific labels
            head_labels = labels[:, i]
            
            # Skip neutral labels (0.5) in loss computation
            valid_mask = head_labels != 0.5
            if valid_mask.any():
                # Ensure consistent dimensions - flatten and ensure at least 1D
                logits_valid = logits[valid_mask].flatten()
                labels_valid = head_labels[valid_mask].flatten()
                
                # Ensure both have the same shape
                if logits_valid.shape != labels_valid.shape:
                    logger.warning(f"Shape mismatch for {head_name}: logits {logits_valid.shape}, labels {labels_valid.shape}")
                    continue
                
                head_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits_valid, 
                    labels_valid
                )
                losses[f"{head_name}_loss"] = head_loss
                total_loss += head_loss
        
        # Compute combined loss using learnable weights
        combined_reward_pos = self.forward(img_pos)
        combined_reward_neg = self.forward(img_neg)
        combined_logits = combined_reward_pos - combined_reward_neg
        
        # Use average of head labels for combined loss
        combined_labels = labels.mean(dim=1)
        combined_loss = nn.functional.binary_cross_entropy_with_logits(
            combined_logits.view(-1), combined_labels.view(-1)
        )
        
        losses["combined_loss"] = combined_loss
        losses["total_loss"] = total_loss + combined_loss
        
        return losses
    
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save model weights.
        
        Parameters
        ----------
        save_directory : str
            Directory to save model weights
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save only the trainable parts (heads and weights)
        state_dict = {
            "heads": self.heads.state_dict(),
            "head_weights": self.head_weights,
        }
        
        save_file(state_dict, save_path / "reward_model.safetensors")
        
        # Save config
        config = {
            "hidden_dim": list(self.heads.values())[0].layers[0].in_features,
            "dropout": 0.1,  # Default value
            "head_names": list(self.heads.keys()),
        }
        
        torch.save(config, save_path / "config.pt")
        logger.info(f"Saved reward model to {save_path}")
    
    @classmethod
    def from_pretrained(cls, model_directory: str, **kwargs: Any) -> "MultiHeadReward":
        """
        Load pretrained model.
        
        Parameters
        ----------
        model_directory : str
            Directory containing saved model
        **kwargs : Any
            Additional arguments for model initialization
            
        Returns
        -------
        MultiHeadReward
            Loaded model
        """
        model_path = Path(model_directory)
        
        # Load config
        config = torch.load(model_path / "config.pt", map_location="cpu")
        
        # Create model
        model = cls(**kwargs)
        
        # Load weights
        state_dict = load_file(model_path / "reward_model.safetensors")
        model.heads.load_state_dict(state_dict["heads"])
        model.head_weights.data = state_dict["head_weights"]
        
        logger.info(f"Loaded reward model from {model_path}")
        return model
    
    def get_head_importance(self) -> dict[str, float]:
        """
        Get normalized importance weights for each head.
        
        Returns
        -------
        dict[str, float]
            Head names and their importance weights
        """
        weights = torch.softmax(self.head_weights, dim=0)
        head_names = list(self.heads.keys())
        
        return {name: weight.item() for name, weight in zip(head_names, weights)}
