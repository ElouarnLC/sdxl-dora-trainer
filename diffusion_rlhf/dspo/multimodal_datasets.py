"""
Enhanced multimodal dataset classes for preference learning in diffusion models.

This module implements datasets that load both images and text prompts
for training multimodal reward models and DSPO fine-tuning.
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


class MultimodalPairPreferenceDataset(Dataset):
    """
    Dataset for loading multimodal preference pairs (image + text) for reward model training.
    
    Loads data from pairs.jsonl with format:
    {"prompt": str, "img_pos": str, "img_neg": str, "label": float}
    
    Also loads actual prompt text from prompts.csv with format:
    {"prompt_id": int, "prompt": str}
    
    Parameters
    ----------
    pairs_file : str
        Path to pairs.jsonl file
    prompts_file : str
        Path to prompts.csv file containing actual prompt text
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
        prompts_file: str,
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
        val_split: float = 0.2,
    ) -> None:
        self.pairs_file = Path(pairs_file)
        self.prompts_file = Path(prompts_file)
        self.image_size = image_size
        self.split = split
        self.val_split = val_split
        
        if not self.pairs_file.exists():
            raise FileNotFoundError(f"Pairs file not found: {self.pairs_file}")
        if not self.prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_file}")
        
        # Load prompts mapping
        self.prompts_map = self._load_prompts()
        
        # Load and split data
        self.pairs = self._load_pairs()
        self.pairs = self._split_data()
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
            
        logger.info(f"Loaded {len(self.pairs)} multimodal pairs for {split} split")
        logger.info(f"Available prompts: {len(self.prompts_map)}")
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from CSV file."""
        prompts_df = pd.read_csv(self.prompts_file)
        
        # Create mapping from prompt_id to prompt text
        prompts_map = {}
        for _, row in prompts_df.iterrows():
            prompts_map[str(row["prompt_id"])] = row["prompt"]
        
        return prompts_map
    
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
    
    def _get_prompt_text(self, prompt_key: str) -> str:
        """Get actual prompt text from prompt key."""
        # Handle different prompt key formats
        if prompt_key.startswith("prompt_"):
            prompt_id = prompt_key.replace("prompt_", "")
        else:
            prompt_id = str(prompt_key)
        
        # Return prompt text or fallback
        return self.prompts_map.get(prompt_id, f"Unknown prompt {prompt_id}")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a multimodal preference pair sample.
        
        Parameters
        ----------
        idx : int
            Sample index
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - prompt: actual prompt text (str)
            - img_pos: positive image tensor
            - img_neg: negative image tensor  
            - label: preference label (1.0 for pos > neg, 0.0 for neg > pos)
        """
        pair = self.pairs[idx]
        
        # Load images
        img_pos = self._load_image(pair["img_pos"])
        img_neg = self._load_image(pair["img_neg"])
        
        # Get actual prompt text
        prompt_text = self._get_prompt_text(pair["prompt"])
        
        # Convert label to tensor
        label = torch.tensor(pair["label"], dtype=torch.float32)
        
        return {
            "prompt": prompt_text,
            "img_pos": img_pos,
            "img_neg": img_neg,
            "label": label,
        }


class MultimodalMultiHeadPairPreferenceDataset(Dataset):
    """
    Dataset for loading multimodal multi-head preference pairs for reward model training.
    
    Expects ratings.csv with columns:
    - prompt_id: Prompt identifier
    - image_path: Path to image
    - spatial_rating: Spatial composition rating (0-10)
    - icono_rating: Iconographic elements rating (0-10)
    - style_rating: Style and aesthetics rating (0-10)
    - fidelity_rating: Image quality rating (0-10)
    - material_rating: Material properties rating (0-10)
    
    Also loads actual prompt text from prompts.csv
    
    Creates preference pairs by comparing ratings across different images
    for the same prompt.
    """
    
    def __init__(
        self,
        ratings_file: str,
        prompts_file: str,
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
        val_split: float = 0.2,
        min_rating_diff: float = 0.5,
    ) -> None:
        """
        Initialize multimodal multi-head preference dataset.
        
        Parameters
        ----------
        ratings_file : str
            Path to ratings CSV file
        prompts_file : str
            Path to prompts.csv file containing actual prompt text
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
        self.prompts_file = Path(prompts_file)
        self.image_size = image_size
        self.split = split
        self.val_split = val_split
        self.min_rating_diff = min_rating_diff
        
        if not self.ratings_file.exists():
            raise FileNotFoundError(f"Ratings file not found: {self.ratings_file}")
        if not self.prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_file}")
        
        # Load prompts mapping
        self.prompts_map = self._load_prompts()
        
        # Load ratings and create pairs
        self.ratings_df = self._load_ratings()
        self.pairs = self._create_preference_pairs()
        self.pairs = self._split_data()
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
            
        logger.info(f"Loaded {len(self.pairs)} multimodal multi-head pairs for {split} split")
        logger.info(f"Available prompts: {len(self.prompts_map)}")
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from CSV file."""
        prompts_df = pd.read_csv(self.prompts_file)
        
        # Create mapping from prompt_id to prompt text
        prompts_map = {}
        for _, row in prompts_df.iterrows():
            prompts_map[str(row["prompt_id"])] = row["prompt"]
        
        return prompts_map
    
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
                # Image paths in CSV are like "data/images/0/1.png"
                # When script runs from diffusion_rlhf/, we just use the path as-is
                image_path = Path(image_path)
            
            image = Image.open(image_path).convert("RGB")
            return self.transform(image)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return black image as fallback
            return torch.zeros(3, self.image_size, self.image_size)
    
    def _get_prompt_text(self, prompt_id: str) -> str:
        """Get actual prompt text from prompt ID."""
        return self.prompts_map.get(str(prompt_id), f"Unknown prompt {prompt_id}")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a multimodal multi-head preference pair sample.
        
        Parameters
        ----------
        idx : int
            Sample index
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - prompt: actual prompt text (str)
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
        
        # Get actual prompt text
        prompt_text = self._get_prompt_text(pair["prompt_id"])
        
        # Convert labels and ratings to tensors
        head_names = ["spatial", "icono", "style", "fidelity", "material"]
        labels = torch.tensor([pair["labels"][head] for head in head_names], dtype=torch.float32)
        ratings_pos = torch.tensor([pair["ratings_pos"][head] for head in head_names], dtype=torch.float32)
        ratings_neg = torch.tensor([pair["ratings_neg"][head] for head in head_names], dtype=torch.float32)
        
        return {
            "prompt": prompt_text,
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
            "unique_prompts": len(set(pair["prompt_id"] for pair in self.pairs)),
        }
        
        for head in head_names:
            labels = [pair["labels"][head] for pair in self.pairs]
            stats["head_preferences"][head] = {
                "positive": sum(1 for l in labels if l == 1.0),
                "negative": sum(1 for l in labels if l == 0.0),
                "neutral": sum(1 for l in labels if l == 0.5),
            }
        
        return stats
