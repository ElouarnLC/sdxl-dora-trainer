"""
Multimodal multi-head reward model for image preference learning.

This module implements a reward model with frozen OpenCLIP ViT-L/14 backbone
that processes both images and text prompts, with 5 specialized MLP heads
for different aspects of image quality and prompt-image alignment.
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


class MultimodalMultiHeadReward(nn.Module):
    """
    Multimodal multi-head reward model with frozen OpenCLIP backbone.
    
    Uses ViT-L/14 as frozen feature extractor for both images and text,
    with 5 specialized heads for different aspects:
    - spatial: Spatial composition and layout
    - icono: Iconographic elements and symbols  
    - style: Artistic style and aesthetics
    - fidelity: Image quality and fidelity
    - material: Material properties and textures
    
    The model processes both images and text prompts to better assess
    prompt-image alignment, which is critical for prompt-conditioned generation.
    
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
    fusion_method : str, default="concat"
        Method to fuse image and text features ("concat", "add", "cross_attention")
    """
    
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        fusion_method: str = "concat",
    ) -> None:
        super().__init__()
        
        # Load OpenCLIP model
        self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        
        # Get tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        # Get feature dimensions
        image_feature_dim = self.backbone.visual.output_dim
        # For OpenCLIP, we can get text feature dim from the text projection shape
        # or use a dummy forward pass to get the actual dimensions
        if hasattr(self.backbone, 'text_projection') and hasattr(self.backbone.text_projection, 'shape'):
            text_feature_dim = self.backbone.text_projection.shape[0]
        else:
            # Alternative: run a dummy forward pass to get the actual dimensions
            with torch.no_grad():
                dummy_text = self.tokenizer(["dummy text"]).to(next(self.backbone.parameters()).device)
                dummy_features = self.backbone.encode_text(dummy_text)
                text_feature_dim = dummy_features.shape[-1]
        
        # Set up fusion
        self.fusion_method = fusion_method
        if fusion_method == "concat":
            fused_dim = image_feature_dim + text_feature_dim
        elif fusion_method == "add":
            if image_feature_dim != text_feature_dim:
                raise ValueError(f"For 'add' fusion, image and text feature dims must match. "
                               f"Got {image_feature_dim} and {text_feature_dim}")
            fused_dim = image_feature_dim
        elif fusion_method == "cross_attention":
            # Use cross-attention to fuse features
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=image_feature_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            # Project text features to match image feature dimension if needed
            if text_feature_dim != image_feature_dim:
                self.text_projection = nn.Linear(text_feature_dim, image_feature_dim)
            else:
                self.text_projection = nn.Identity()
            fused_dim = image_feature_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Create specialized heads
        self.heads = nn.ModuleDict({
            "spatial": MLPHead(fused_dim, hidden_dim, 1, dropout),
            "icono": MLPHead(fused_dim, hidden_dim, 1, dropout),
            "style": MLPHead(fused_dim, hidden_dim, 1, dropout),
            "fidelity": MLPHead(fused_dim, hidden_dim, 1, dropout),
            "material": MLPHead(fused_dim, hidden_dim, 1, dropout),
        })
        
        # Learnable weights for combining heads
        self.head_weights = nn.Parameter(torch.ones(5) / 5)
        
        logger.info(f"Initialized MultimodalMultiHeadReward with {model_name} backbone")
        logger.info(f"Image feature dimension: {image_feature_dim}")
        logger.info(f"Text feature dimension: {text_feature_dim}")
        logger.info(f"Fusion method: {fusion_method}")
        logger.info(f"Fused feature dimension: {fused_dim}")
        logger.info(f"Backbone frozen: {freeze_backbone}")
    
    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images using frozen backbone.
        
        Parameters
        ----------
        images : torch.Tensor
            Batch of images [B, C, H, W]
            
        Returns
        -------
        torch.Tensor
            Image features [B, image_feature_dim]
        """
        with torch.set_grad_enabled(self.training and not self._backbone_frozen()):
            features = self.backbone.encode_image(images)
            
        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def extract_text_features(self, prompts: list[str]) -> torch.Tensor:
        """
        Extract features from text prompts using frozen backbone.
        
        Parameters
        ----------
        prompts : list[str]
            List of text prompts
            
        Returns
        -------
        torch.Tensor
            Text features [B, text_feature_dim]
        """
        # Tokenize prompts
        text_tokens = self.tokenizer(prompts).to(next(self.parameters()).device)
        
        with torch.set_grad_enabled(self.training and not self._backbone_frozen()):
            features = self.backbone.encode_text(text_tokens)
            
        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def fuse_features(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse image and text features.
        
        Parameters
        ----------
        image_features : torch.Tensor
            Image features [B, image_feature_dim]
        text_features : torch.Tensor
            Text features [B, text_feature_dim]
            
        Returns
        -------
        torch.Tensor
            Fused features [B, fused_dim]
        """
        if self.fusion_method == "concat":
            return torch.cat([image_features, text_features], dim=-1)
        
        elif self.fusion_method == "add":
            return image_features + text_features
        
        elif self.fusion_method == "cross_attention":
            # Project text features if needed
            text_features_proj = self.text_projection(text_features)
            
            # Use text as query, image as key/value for cross-attention
            # Add sequence dimension for attention
            text_query = text_features_proj.unsqueeze(1)  # [B, 1, dim]
            image_kv = image_features.unsqueeze(1)        # [B, 1, dim]
            
            # Apply cross-attention
            attended_features, _ = self.cross_attention(
                query=text_query,
                key=image_kv,
                value=image_kv
            )
            
            # Combine attended text with original image features
            fused = image_features + attended_features.squeeze(1)
            return fused
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _backbone_frozen(self) -> bool:
        """Check if backbone is frozen."""
        return not next(self.backbone.parameters()).requires_grad
    
    def forward(
        self, 
        images: torch.Tensor,
        prompts: list[str],
        return_individual: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass through multimodal reward model.
        
        Parameters
        ----------
        images : torch.Tensor
            Batch of images [B, C, H, W]
        prompts : list[str]
            List of text prompts
        return_individual : bool, default=False
            Whether to return individual head outputs
            
        Returns
        -------
        torch.Tensor or dict
            If return_individual=False: Combined reward scores [B, 1]
            If return_individual=True: Dict with individual head outputs
        """
        # Extract features
        image_features = self.extract_image_features(images)
        text_features = self.extract_text_features(prompts)
        
        # Fuse features
        fused_features = self.fuse_features(image_features, text_features)
        
        # Get individual head outputs
        head_outputs = {}
        for name, head in self.heads.items():
            head_outputs[name] = head(fused_features)
        
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
        prompts: list[str],
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
        prompts : list[str]
            List of text prompts (same for both images)
        labels : torch.Tensor
            Preference labels [B] or [B, 5] for multi-head
            
        Returns
        -------
        torch.Tensor
            Preference loss scalar
        """
        # Get reward scores
        reward_pos = self.forward(img_pos, prompts)
        reward_neg = self.forward(img_neg, prompts)
        
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
        prompts: list[str],
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
        prompts : list[str]
            List of text prompts (same for both images)
        labels : torch.Tensor
            Multi-head preference labels [B, 5]
            
        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with individual head losses and total loss
        """
        # Get individual head outputs
        head_outputs_pos = self.forward(img_pos, prompts, return_individual=True)
        head_outputs_neg = self.forward(img_neg, prompts, return_individual=True)
        
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
        combined_reward_pos = self.forward(img_pos, prompts)
        combined_reward_neg = self.forward(img_neg, prompts)
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
        
        # Save only the trainable parts (heads, weights, and fusion components)
        state_dict = {}
        
        # Save heads
        heads_state_dict = self.heads.state_dict()
        for key, value in heads_state_dict.items():
            state_dict[f"heads.{key}"] = value
        
        # Save head weights
        state_dict["head_weights"] = self.head_weights
        
        # Save fusion components if they exist
        if hasattr(self, 'cross_attention'):
            cross_attn_state_dict = self.cross_attention.state_dict()
            for key, value in cross_attn_state_dict.items():
                state_dict[f"cross_attention.{key}"] = value
        
        if hasattr(self, 'text_projection'):
            text_proj_state_dict = self.text_projection.state_dict()
            for key, value in text_proj_state_dict.items():
                state_dict[f"text_projection.{key}"] = value
        
        save_file(state_dict, save_path / "multimodal_reward_model.safetensors")
        
        # Save config
        config = {
            "hidden_dim": list(self.heads.values())[0].layers[0].in_features,
            "dropout": 0.1,  # Default value
            "fusion_method": self.fusion_method,
            "head_names": list(self.heads.keys()),
        }
        
        torch.save(config, save_path / "config.pt")
        logger.info(f"Saved multimodal reward model to {save_path}")
    
    @classmethod
    def from_pretrained(
        cls, 
        model_directory: str, 
        **kwargs: Any
    ) -> "MultimodalMultiHeadReward":
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
        MultimodalMultiHeadReward
            Loaded model
        """
        model_path = Path(model_directory)
        
        # Load config
        config = torch.load(model_path / "config.pt", map_location="cpu")
        
        # Create model with config parameters
        fusion_method = config.get("fusion_method", "concat")
        model = cls(fusion_method=fusion_method, **kwargs)
        
        # Load weights
        state_dict = load_file(model_path / "multimodal_reward_model.safetensors")
        
        # Reconstruct heads state dict from flattened format
        heads_state_dict = {}
        cross_attn_state_dict = {}
        text_proj_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith("heads."):
                heads_key = key[6:]  # Remove "heads." prefix
                heads_state_dict[heads_key] = value
            elif key.startswith("cross_attention."):
                cross_attn_key = key[17:]  # Remove "cross_attention." prefix
                cross_attn_state_dict[cross_attn_key] = value
            elif key.startswith("text_projection."):
                text_proj_key = key[16:]  # Remove "text_projection." prefix
                text_proj_state_dict[text_proj_key] = value
        
        # Load state dicts
        model.heads.load_state_dict(heads_state_dict)
        model.head_weights.data = state_dict["head_weights"]
        
        if cross_attn_state_dict and hasattr(model, 'cross_attention'):
            model.cross_attention.load_state_dict(cross_attn_state_dict)
        
        if text_proj_state_dict and hasattr(model, 'text_projection'):
            model.text_projection.load_state_dict(text_proj_state_dict)
        
        logger.info(f"Loaded multimodal reward model from {model_path}")
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
        
        return {
            name: weight.item()
            for name, weight in zip(head_names, weights, strict=True)
        }
