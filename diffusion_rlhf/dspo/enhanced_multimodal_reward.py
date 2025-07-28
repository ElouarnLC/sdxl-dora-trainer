#!/usr/bin/env python3
"""
Enhanced multimodal reward model with improved architecture and training strategies.

Key improvements:
1. Better backbone models (CLIP ViT-L/14-336 or ViT-g-14)
2. Improved fusion mechanisms with attention and residual connections
3. Temperature-scaled contrastive learning
4. Advanced loss functions (focal loss, label smoothing)
5. Data augmentation strategies
6. Better regularization techniques
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in preference learning.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Weighting factor for rare class
    gamma : float, default=2.0
        Focusing parameter to down-weight easy examples
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingBCE(nn.Module):
    """
    Label smoothing for binary classification to prevent overconfidence.
    
    Parameters
    ----------
    smoothing : float, default=0.1
        Label smoothing factor
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed BCE loss."""
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(inputs, targets_smooth)


class ImprovedMLPHead(nn.Module):
    """
    Enhanced MLP head with better architecture and regularization.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dims : List[int], default=[512, 256]
        Hidden layer dimensions
    output_dim : int, default=1
        Output dimension
    dropout : float, default=0.1
        Dropout probability
    use_batch_norm : bool, default=True
        Whether to use batch normalization
    activation : str, default="gelu"
        Activation function
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 1,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "swish":
                layers.append(nn.SiLU())
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through enhanced MLP head."""
        return self.layers(x)


class AttentionFusion(nn.Module):
    """
    Enhanced attention-based fusion mechanism for image and text features.
    
    Parameters
    ----------
    image_dim : int
        Image feature dimension
    text_dim : int
        Text feature dimension
    hidden_dim : int, default=512
        Hidden dimension for attention
    num_heads : int, default=8
        Number of attention heads
    dropout : float, default=0.1
        Dropout probability
    """
    
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Project features to common dimension
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention for refinement
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Gate mechanism for adaptive fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse image and text features using attention mechanism.
        
        Parameters
        ----------
        image_features : torch.Tensor
            Image features [B, image_dim]
        text_features : torch.Tensor
            Text features [B, text_dim]
            
        Returns
        -------
        torch.Tensor
            Fused features [B, hidden_dim]
        """
        batch_size = image_features.size(0)
        
        # Project to common dimension
        img_proj = self.image_proj(image_features)  # [B, hidden_dim]
        txt_proj = self.text_proj(text_features)    # [B, hidden_dim]
        
        # Add sequence dimension for attention
        img_seq = img_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        txt_seq = txt_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Cross-attention: image attends to text
        img_attended, _ = self.cross_attention(
            query=img_seq,
            key=txt_seq,
            value=txt_seq
        )
        img_attended = self.norm1(img_attended + img_seq)
        
        # Cross-attention: text attends to image
        txt_attended, _ = self.cross_attention(
            query=txt_seq,
            key=img_seq,
            value=img_seq
        )
        txt_attended = self.norm2(txt_attended + txt_seq)
        
        # Concatenate attended features
        combined = torch.cat([img_attended, txt_attended], dim=1)  # [B, 2, hidden_dim]
        
        # Self-attention for interaction modeling
        refined, _ = self.self_attention(combined, combined, combined)
        refined = self.norm3(refined + combined)
        
        # Feed-forward network
        refined = refined + self.ffn(refined)
        
        # Pool to single vector
        pooled = refined.mean(dim=1)  # [B, hidden_dim]
        
        # Adaptive gating
        gate_input = torch.cat([img_proj, txt_proj], dim=-1)
        gate_weights = self.gate(gate_input)
        
        # Apply gating to balance image vs text importance
        img_gated = gate_weights * img_proj
        txt_gated = (1 - gate_weights) * txt_proj
        
        # Final fusion
        fused = pooled + img_gated + txt_gated
        
        return fused


class TemperatureScaledContrastiveLoss(nn.Module):
    """
    Temperature-scaled contrastive loss for better preference learning.
    
    Parameters
    ----------
    temperature : float, default=0.07
        Temperature parameter for scaling
    margin : float, default=0.2
        Margin for contrastive learning
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self, 
        reward_pos: torch.Tensor, 
        reward_neg: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temperature-scaled contrastive loss.
        
        Parameters
        ----------
        reward_pos : torch.Tensor
            Rewards for positive examples
        reward_neg : torch.Tensor
            Rewards for negative examples
        labels : torch.Tensor
            Preference labels
            
        Returns
        -------
        torch.Tensor
            Contrastive loss
        """
        # Scale by temperature
        reward_pos_scaled = reward_pos / self.temperature
        reward_neg_scaled = reward_neg / self.temperature
        
        # Compute margin-based contrastive loss
        diff = reward_pos_scaled - reward_neg_scaled
        
        # Positive pairs: minimize distance
        pos_loss = torch.clamp(self.margin - diff, min=0.0) ** 2
        
        # Negative pairs: maximize distance  
        neg_loss = torch.clamp(diff + self.margin, min=0.0) ** 2
        
        # Combine based on labels
        loss = labels * pos_loss + (1 - labels) * neg_loss
        
        return loss.mean()


class EnhancedMultimodalMultiHeadReward(nn.Module):
    """
    Enhanced multimodal multi-head reward model with state-of-the-art improvements.
    
    Key improvements:
    1. Better backbone models (CLIP ViT-L/14-336 or ConvNeXT)
    2. Enhanced fusion mechanisms with attention
    3. Improved MLP heads with residual connections
    4. Advanced loss functions (focal loss, contrastive learning)
    5. Better regularization and normalization
    
    Parameters
    ----------
    model_name : str, default="ViT-L-14-336"
        OpenCLIP model name (ViT-L-14-336 for better resolution)
    pretrained : str, default="openai"
        Pretrained weights dataset
    hidden_dims : List[int], default=[512, 256]
        Hidden dimensions for MLP heads
    dropout : float, default=0.15
        Dropout probability
    freeze_backbone : bool, default=True
        Whether to freeze the backbone
    fusion_method : str, default="attention"
        Fusion method: "attention", "concat", "add", "cross_attention"
    loss_type : str, default="focal"
        Loss function: "bce", "focal", "label_smooth", "contrastive"
    temperature : float, default=0.07
        Temperature for contrastive loss
    use_scheduler : bool, default=True
        Whether to use learning rate scheduling
    """
    
    def __init__(
        self,
        model_name: str = "ViT-L-14-336",  # Higher resolution model
        pretrained: str = "openai",
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.15,
        freeze_backbone: bool = True,
        fusion_method: str = "attention",
        loss_type: str = "focal",
        temperature: float = 0.07,
        use_scheduler: bool = True,
    ) -> None:
        super().__init__()
        
        self.loss_type = loss_type
        self.temperature = temperature
        self.use_scheduler = use_scheduler
        
        # Load better backbone model
        try:
            self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
        except:
            # Fallback to standard model if high-res not available
            logger.warning(f"Failed to load {model_name}, falling back to ViT-L-14")
            self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="laion2b_s32b_b82k"
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
        
        # Get text feature dimension
        with torch.no_grad():
            dummy_text = self.tokenizer(["dummy text"]).to(next(self.backbone.parameters()).device)
            dummy_features = self.backbone.encode_text(dummy_text)
            text_feature_dim = dummy_features.shape[-1]
        
        # Set up enhanced fusion
        self.fusion_method = fusion_method
        if fusion_method == "attention":
            self.fusion = AttentionFusion(
                image_dim=image_feature_dim,
                text_dim=text_feature_dim,
                hidden_dim=max(hidden_dims),
                dropout=dropout
            )
            fused_dim = max(hidden_dims)
        elif fusion_method == "concat":
            fused_dim = image_feature_dim + text_feature_dim
        elif fusion_method == "add":
            if image_feature_dim != text_feature_dim:
                self.text_projection = nn.Linear(text_feature_dim, image_feature_dim)
            else:
                self.text_projection = nn.Identity()
            fused_dim = image_feature_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Create enhanced heads with better architecture
        self.heads = nn.ModuleDict({
            "spatial": ImprovedMLPHead(fused_dim, hidden_dims, 1, dropout),
            "icono": ImprovedMLPHead(fused_dim, hidden_dims, 1, dropout),
            "style": ImprovedMLPHead(fused_dim, hidden_dims, 1, dropout),
            "fidelity": ImprovedMLPHead(fused_dim, hidden_dims, 1, dropout),
            "material": ImprovedMLPHead(fused_dim, hidden_dims, 1, dropout),
        })
        
        # Enhanced learnable weights with better initialization
        self.head_weights = nn.Parameter(torch.ones(5) / 5)
        
        # Loss functions
        if loss_type == "focal":
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif loss_type == "label_smooth":
            self.criterion = LabelSmoothingBCE(smoothing=0.1)
        elif loss_type == "contrastive":
            self.criterion = TemperatureScaledContrastiveLoss(temperature=temperature)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        logger.info(f"Enhanced multimodal reward model initialized:")
        logger.info(f"  Backbone: {model_name}")
        logger.info(f"  Image dim: {image_feature_dim}, Text dim: {text_feature_dim}")
        logger.info(f"  Fusion: {fusion_method}, Fused dim: {fused_dim}")
        logger.info(f"  Loss type: {loss_type}")
        logger.info(f"  Hidden dims: {hidden_dims}")
    
    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract enhanced image features."""
        with torch.set_grad_enabled(self.training and not self._backbone_frozen()):
            features = self.backbone.encode_image(images)
        
        # L2 normalization for better stability
        features = F.normalize(features, p=2, dim=-1)
        return features
    
    def extract_text_features(self, prompts: List[str]) -> torch.Tensor:
        """Extract enhanced text features."""
        text_tokens = self.tokenizer(prompts).to(next(self.parameters()).device)
        
        with torch.set_grad_enabled(self.training and not self._backbone_frozen()):
            features = self.backbone.encode_text(text_tokens)
        
        # L2 normalization for better stability
        features = F.normalize(features, p=2, dim=-1)
        return features
    
    def fuse_features(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """Enhanced feature fusion."""
        if self.fusion_method == "attention":
            return self.fusion(image_features, text_features)
        elif self.fusion_method == "concat":
            return torch.cat([image_features, text_features], dim=-1)
        elif self.fusion_method == "add":
            text_proj = self.text_projection(text_features)
            return image_features + text_proj
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def forward(
        self, 
        images: torch.Tensor, 
        prompts: List[str], 
        return_individual: bool = False
    ) -> torch.Tensor:
        """Enhanced forward pass."""
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
        
        # Enhanced combination with temperature scaling
        weights = F.softmax(self.head_weights / self.temperature, dim=0)
        
        weighted_outputs = []
        for i, (name, output) in enumerate(head_outputs.items()):
            weighted_outputs.append(weights[i] * output)
        
        combined_reward = torch.stack(weighted_outputs, dim=-1).sum(dim=-1)
        return combined_reward
    
    def compute_enhanced_loss(
        self,
        img_pos: torch.Tensor,
        img_neg: torch.Tensor,
        prompts: List[str],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute enhanced loss with better training dynamics."""
        # Get reward scores
        reward_pos = self.forward(img_pos, prompts)
        reward_neg = self.forward(img_neg, prompts)
        
        if self.loss_type == "contrastive":
            return self.criterion(reward_pos, reward_neg, labels)
        else:
            # Standard preference learning
            logits = reward_pos - reward_neg
            return self.criterion(logits.squeeze(), labels)
    
    def _backbone_frozen(self) -> bool:
        """Check if backbone is frozen."""
        return not next(self.backbone.parameters()).requires_grad
    
    def get_enhanced_metrics(
        self,
        img_pos: torch.Tensor,
        img_neg: torch.Tensor,
        prompts: List[str],
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Get enhanced evaluation metrics."""
        with torch.no_grad():
            reward_pos = self.forward(img_pos, prompts)
            reward_neg = self.forward(img_neg, prompts)
            
            # Basic metrics
            logits = reward_pos - reward_neg
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            
            accuracy = (predictions.squeeze() == labels).float().mean().item()
            
            # Confidence-based metrics
            confidence = torch.abs(probs - 0.5) * 2  # Scale to [0, 1]
            avg_confidence = confidence.mean().item()
            
            # Margin-based metrics
            margin = torch.abs(reward_pos - reward_neg).mean().item()
            
            return {
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "reward_margin": margin,
            }


def create_enhanced_training_config() -> Dict[str, Any]:
    """Create enhanced training configuration with better hyperparameters."""
    return {
        # Model architecture
        "model_name": "ViT-L-14-336",  # Higher resolution for better performance
        "fusion_method": "attention",   # Advanced fusion
        "loss_type": "focal",          # Better handling of imbalanced data
        "hidden_dims": [768, 384, 192], # Deeper network
        "dropout": 0.15,               # Moderate regularization
        
        # Training dynamics
        "learning_rate": 5e-5,         # Lower LR for stability
        "weight_decay": 0.02,          # L2 regularization
        "batch_size": 16,              # Larger batches for stability
        "warmup_steps": 500,           # Learning rate warmup
        "max_grad_norm": 1.0,          # Gradient clipping
        
        # Data augmentation
        "mixup_alpha": 0.2,            # Mixup augmentation
        "cutmix_alpha": 1.0,           # CutMix augmentation
        "label_smoothing": 0.1,        # Label smoothing
        
        # Regularization
        "temperature": 0.05,           # Temperature scaling
        "margin": 0.3,                 # Contrastive margin
        "ema_decay": 0.999,            # Exponential moving average
        
        # Optimization
        "scheduler": "cosine_with_warmup",
        "min_lr_ratio": 0.01,
        "num_cycles": 1,
    }
