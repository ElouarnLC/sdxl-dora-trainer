"""
Enhanced dataset class for multi-head preference learning.

This module extends the PairPreferenceDataset to handle multi-head ratings
for spatial, iconographic, style, fidelity, and material aspects.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jsonlines
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class MultiHeadPairPreferenceDataset(Dataset):
    """
    Dataset for loading multi-head preference pairs for reward model training.
    
    Expects ratings.csv with columns:
    - prompt_id: Prompt identifier
    - image_path: Path to image
    - spatial_rating: Spatial composition rating (0-10)
    - icono_rating: Iconographic elements rating (0-10)
    - style_rating: Style and aesthetics rating (0-10)
    - fidelity_rating: Image quality rating (0-10)
    - material_rating: Material properties rating (0-10)
    
    Creates preference pairs by comparing ratings across different images
    for the same prompt.
    """
    
    def __init__(
        self,
        ratings_file: str,
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
        val_split: float = 0.2,
        min_rating_diff: float = 0.5,
    ) -> None:
        """
        Initialize multi-head preference dataset.
        
        Parameters
        ----------
        ratings_file : str
            Path to ratings CSV file
        image_size : int, default=224
            Size to resize images to
        transform : Optional[transforms.Compose], default=None
            Custom image transforms
        split : str, default="train"
            Dataset split ("train" or "val")
        val_split : float, default=0.2
            Validation split ratio
        min_rating_diff : float, default=0.5
            Minimum rating difference to create a preference pair
        """
        self.ratings_file = Path(ratings_file)
        self.image_size = image_size
        self.split = split
        self.val_split = val_split
        self.min_rating_diff = min_rating_diff
        
        if not self.ratings_file.exists():
            raise FileNotFoundError(f"Ratings file not found: {self.ratings_file}")
        
        # Load ratings and create pairs
        self.ratings_df = self._load_ratings()
        self.pairs = self._create_preference_pairs()
        self.pairs = self._split_data()
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
            
        logger.info(f"Loaded {len(self.pairs)} multi-head pairs for {split} split")
    
    def _load_ratings(self) -> pd.DataFrame:
        """Load ratings from CSV file."""
        expected_columns = [
            "prompt_id", "image_path", "spatial_rating", "icono_rating",
            "style_rating", "fidelity_rating", "material_rating"
        ]
        
        df = pd.read_csv(self.ratings_file)
        
        # Check required columns
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in ratings file: {missing_cols}")
        
        return df
    
    def _create_preference_pairs(self) -> List[Dict]:
        """Create preference pairs from ratings."""
        pairs = []
        head_names = ["spatial", "icono", "style", "fidelity", "material"]
        
        # Group by prompt_id
        for prompt_id, group in self.ratings_df.groupby("prompt_id"):
            if len(group) < 2:
                continue
            
            # Create all possible pairs within this prompt
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    img1 = group.iloc[i]
                    img2 = group.iloc[j]
                    
                    # Calculate ratings for each head
                    ratings1 = {head: img1[f"{head}_rating"] for head in head_names}
                    ratings2 = {head: img2[f"{head}_rating"] for head in head_names}
                    
                    # Create preference labels for each head
                    labels = {}
                    valid_pair = False
                    
                    for head in head_names:
                        diff = ratings1[head] - ratings2[head]
                        if abs(diff) >= self.min_rating_diff:
                            labels[head] = 1.0 if diff > 0 else 0.0
                            valid_pair = True
                        else:
                            labels[head] = 0.5  # Neutral/ambiguous
                    
                    if valid_pair:
                        pairs.append({
                            "prompt_id": prompt_id,
                            "img_pos": img1["image_path"],
                            "img_neg": img2["image_path"],
                            "labels": labels,
                            "ratings_pos": ratings1,
                            "ratings_neg": ratings2,
                        })
        
        return pairs
    
    def _split_data(self) -> List[Dict]:
        """Split data into train/val based on split parameter."""
        n_val = int(len(self.pairs) * self.val_split)
        
        if self.split == "train":
            return self.pairs[n_val:]
        elif self.split == "val":
            return self.pairs[:n_val]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image preprocessing transforms."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        try:
            # Handle relative paths
            if not Path(image_path).is_absolute():
                image_path = self.ratings_file.parent / image_path
            
            image = Image.open(image_path).convert("RGB")
            return self.transform(image)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return black image as fallback
            return torch.zeros(3, self.image_size, self.image_size)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a multi-head preference pair sample.
        
        Parameters
        ----------
        idx : int
            Sample index
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - img_pos: positive image tensor
            - img_neg: negative image tensor
            - labels: multi-head preference labels [5]
            - ratings_pos: positive image ratings [5]
            - ratings_neg: negative image ratings [5]
        """
        pair = self.pairs[idx]
        
        # Load images
        img_pos = self._load_image(pair["img_pos"])
        img_neg = self._load_image(pair["img_neg"])
        
        # Convert labels and ratings to tensors
        head_names = ["spatial", "icono", "style", "fidelity", "material"]
        labels = torch.tensor([pair["labels"][head] for head in head_names], dtype=torch.float32)
        ratings_pos = torch.tensor([pair["ratings_pos"][head] for head in head_names], dtype=torch.float32)
        ratings_neg = torch.tensor([pair["ratings_neg"][head] for head in head_names], dtype=torch.float32)
        
        return {
            "img_pos": img_pos,
            "img_neg": img_neg,
            "labels": labels,
            "ratings_pos": ratings_pos,
            "ratings_neg": ratings_neg,
        }
    
    def get_head_names(self) -> List[str]:
        """Get list of head names."""
        return ["spatial", "icono", "style", "fidelity", "material"]
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics."""
        if not self.pairs:
            return {}
        
        head_names = self.get_head_names()
        stats = {
            "total_pairs": len(self.pairs),
            "head_preferences": {},
            "rating_distributions": {},
        }
        
        for head in head_names:
            labels = [pair["labels"][head] for pair in self.pairs]
            stats["head_preferences"][head] = {
                "positive": sum(1 for l in labels if l == 1.0),
                "negative": sum(1 for l in labels if l == 0.0),
                "neutral": sum(1 for l in labels if l == 0.5),
            }
        
        return stats
