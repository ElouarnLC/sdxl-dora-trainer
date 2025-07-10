"""
Dataset classes for preference learning in diffusion models.

This module implements PairPreferenceDataset that loads preference pairs
from JSONL files for training reward models and DSPO fine-tuning.
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


class PairPreferenceDataset(Dataset):
    """
    Dataset for loading preference pairs for reward model training and DSPO.
    
    Loads data from pairs.jsonl with format:
    {"prompt": str, "img_pos": str, "img_neg": str, "label": float}
    
    Parameters
    ----------
    pairs_file : str
        Path to pairs.jsonl file
    image_size : int, default=224
        Size to resize images to
    transform : Optional[transforms.Compose], default=None
        Custom image transforms. If None, uses default preprocessing
    split : str, default="train"
        Dataset split for train/val/test
    val_split : float, default=0.2
        Fraction of data to use for validation
    """
    
    def __init__(
        self,
        pairs_file: str,
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
        val_split: float = 0.2,
    ) -> None:
        self.pairs_file = Path(pairs_file)
        self.image_size = image_size
        self.split = split
        self.val_split = val_split
        
        if not self.pairs_file.exists():
            raise FileNotFoundError(f"Pairs file not found: {self.pairs_file}")
        
        # Load and split data
        self.pairs = self._load_pairs()
        self.pairs = self._split_data()
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
            
        logger.info(f"Loaded {len(self.pairs)} pairs for {split} split")
    
    def _load_pairs(self) -> List[Dict]:
        """Load preference pairs from JSONL file."""
        pairs = []
        with jsonlines.open(self.pairs_file, "r") as reader:
            for line in reader:
                pairs.append(line)
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
        Get a preference pair sample.
        
        Parameters
        ----------
        idx : int
            Sample index
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - prompt: tokenized prompt
            - img_pos: positive image tensor
            - img_neg: negative image tensor  
            - label: preference label (1.0 for pos > neg, 0.0 for neg > pos)
        """
        pair = self.pairs[idx]
        
        # Load images
        img_pos = self._load_image(pair["img_pos"])
        img_neg = self._load_image(pair["img_neg"])
        
        # Convert label to tensor
        label = torch.tensor(pair["label"], dtype=torch.float32)
        
        return {
            "prompt": pair["prompt"],
            "img_pos": img_pos,
            "img_neg": img_neg,
            "label": label,
        }
    
    @classmethod
    def from_ratings(
        cls,
        ratings_file: str,
        images_dir: str,
        output_file: str,
        threshold: float = 0.5,
    ) -> "PairPreferenceDataset":
        """
        Create dataset from ratings CSV file.
        
        Parameters
        ----------
        ratings_file : str
            Path to ratings.csv with columns: prompt_id, seed, rating
        images_dir : str
            Directory containing generated images
        output_file : str
            Path to output pairs.jsonl file
        threshold : float, default=0.5
            Minimum rating difference to create a pair
            
        Returns
        -------
        PairPreferenceDataset
            Dataset created from the generated pairs
        """
        # Load ratings
        ratings_df = pd.read_csv(ratings_file)
        
        # Create pairs
        pairs = []
        for prompt_id in ratings_df["prompt_id"].unique():
            prompt_ratings = ratings_df[ratings_df["prompt_id"] == prompt_id]
            
            # Compare all pairs of images for this prompt
            for i, row1 in prompt_ratings.iterrows():
                for j, row2 in prompt_ratings.iterrows():
                    if i >= j:
                        continue
                        
                    rating_diff = abs(row1["rating"] - row2["rating"])
                    if rating_diff < threshold:
                        continue
                    
                    # Determine which is better
                    if row1["rating"] > row2["rating"]:
                        pos_row, neg_row = row1, row2
                        label = 1.0
                    else:
                        pos_row, neg_row = row2, row1
                        label = 0.0
                    
                    # Build image paths
                    img_pos = f"{images_dir}/{pos_row['prompt_id']}/{pos_row['seed']}.png"
                    img_neg = f"{images_dir}/{neg_row['prompt_id']}/{neg_row['seed']}.png"
                    
                    pairs.append({
                        "prompt": f"prompt_{prompt_id}",  # Will be replaced with actual prompt
                        "img_pos": img_pos,
                        "img_neg": img_neg,
                        "label": label,
                    })
        
        # Save pairs to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with jsonlines.open(output_file, "w") as writer:
            for pair in pairs:
                writer.write(pair)
        
        logger.info(f"Created {len(pairs)} preference pairs in {output_file}")
        
        return cls(output_file)
