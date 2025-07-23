#!/usr/bin/env python3
"""
Build preference pairs from human ratings for reward model training.

Converts ratings.csv into pairs.jsonl format for preference optimization.
Each pair contains a prompt and two images with preference labels.
"""

import argparse
import logging
from pathlib import Path

import jsonlines
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_prompts(prompts_file: str) -> dict[int, str]:
    """
    Load prompts from CSV file.
    
    Parameters
    ----------
    prompts_file : str
        Path to prompts.csv file
        
    Returns
    -------
    dict[int, str]
        Mapping from prompt_id to prompt text
    """
    prompts_df = pd.read_csv(prompts_file)
    return dict(zip(prompts_df["prompt_id"], prompts_df["prompt"]))


def build_preference_pairs(
    ratings_df: pd.DataFrame,
    prompts_dict: dict[int, str],
    images_dir: Path,
    threshold: float = 0.5,
) -> list[dict]:
    """
    Build preference pairs from ratings dataframe.
    
    Parameters
    ----------
    ratings_df : pd.DataFrame
        DataFrame with columns: prompt_id, seed, rating
    prompts_dict : dict[int, str]
        Mapping from prompt_id to prompt text
    images_dir : Path
        Directory containing generated images
    threshold : float, default=0.5
        Minimum rating difference to create a pair
        
    Returns
    -------
    list[dict]
        List of preference pairs
    """
    pairs = []
    
    for prompt_id in tqdm(ratings_df["prompt_id"].unique(), desc="Building pairs"):
        prompt_ratings = ratings_df[ratings_df["prompt_id"] == prompt_id]
        
        if len(prompt_ratings) < 2:
            logger.warning(f"Prompt {prompt_id} has fewer than 2 ratings, skipping")
            continue
        
        prompt_text = prompts_dict.get(prompt_id, f"prompt_{prompt_id}")
        
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
                img_pos = images_dir / str(pos_row["prompt_id"]) / f"{pos_row['seed']}.png"
                img_neg = images_dir / str(neg_row["prompt_id"]) / f"{neg_row['seed']}.png"
                
                # Verify images exist
                if not img_pos.exists():
                    logger.warning(f"Positive image not found: {img_pos}")
                    continue
                if not img_neg.exists():
                    logger.warning(f"Negative image not found: {img_neg}")
                    continue
                
                pairs.append({
                    "prompt": prompt_text,
                    "img_pos": str(img_pos),
                    "img_neg": str(img_neg),
                    "label": label,
                    "rating_pos": float(pos_row["rating"]),
                    "rating_neg": float(neg_row["rating"]),
                    "rating_diff": float(rating_diff),
                })
    
    return pairs


def save_pairs(pairs: list[dict], output_file: Path) -> None:
    """
    Save preference pairs to JSONL file.
    
    Parameters
    ----------
    pairs : list[dict]
        List of preference pairs
    output_file : Path
        Output file path
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(output_file, "w") as writer:
        for pair in pairs:
            writer.write(pair)
    
    logger.info(f"Saved {len(pairs)} preference pairs to {output_file}")


def analyze_pairs(pairs: list[dict]) -> None:
    """
    Analyze and log statistics about preference pairs.
    
    Parameters
    ----------
    pairs : list[dict]
        List of preference pairs
    """
    if not pairs:
        logger.warning("No pairs generated")
        return
    
    # Count unique prompts
    unique_prompts = len(set(pair["prompt"] for pair in pairs))
    
    # Rating difference statistics
    rating_diffs = [pair["rating_diff"] for pair in pairs]
    avg_diff = sum(rating_diffs) / len(rating_diffs)
    min_diff = min(rating_diffs)
    max_diff = max(rating_diffs)
    
    # Label distribution
    positive_pairs = sum(1 for pair in pairs if pair["label"] == 1.0)
    negative_pairs = len(pairs) - positive_pairs
    
    logger.info(f"Generated {len(pairs)} preference pairs")
    logger.info(f"Unique prompts: {unique_prompts}")
    logger.info(f"Rating difference - avg: {avg_diff:.2f}, min: {min_diff:.2f}, max: {max_diff:.2f}")
    logger.info(f"Label distribution - positive: {positive_pairs}, negative: {negative_pairs}")


def find_generated_csv() -> str:
    """
    Automatically find the generated CSV file from either single or dual GPU generation.
    
    Returns
    -------
    str
        Path to the generated CSV file
        
    Raises
    ------
    FileNotFoundError
        If no generated CSV file is found
    """
    possible_files = [
        "data/generated_dual_gpu.csv",  # From dual GPU generation
        "data/generated.csv",           # From single GPU generation
    ]
    
    for file_path in possible_files:
        if Path(file_path).exists():
            logger.info(f"Found generated images CSV: {file_path}")
            return file_path
    
    raise FileNotFoundError(
        "No generated CSV file found. Expected one of: " +
        ", ".join(possible_files) +
        ". Please run 'make generate-images-dual' or " +
        "'make generate-images-single' first."
    )


def create_multihead_template(images_csv: str, output_file: Path) -> None:
    """
    Create a multi-head rating template from generated images CSV.
    
    Parameters
    ----------
    images_csv : str
        Path to generated images CSV file
    output_file : Path
        Output path for multihead ratings template
    """
    # Load generated images data
    images_df = pd.read_csv(images_csv)
    
    # Handle different column names for image path
    image_path_col = "image_path" if "image_path" in images_df.columns else "path"
    
    # Create template with multi-head rating columns
    template_rows = []
    for _, row in images_df.iterrows():
        template_rows.append({
            "prompt_id": row["prompt_id"],
            "image_path": row[image_path_col],
            "spatial_rating": "",  # Composition, layout, rule of thirds, balance
            "icono_rating": "",    # Symbols, cultural elements, narrative
            "style_rating": "",    # Artistic technique, color harmony
            "fidelity_rating": "", # Technical quality, sharpness, noise
            "material_rating": "", # Texture realism, surface properties
        })
    
    # Save template
    template_df = pd.DataFrame(template_rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    template_df.to_csv(output_file, index=False)
    
    logger.info(f"Created multi-head rating template with {len(template_rows)} images")
    logger.info(f"Template saved to: {output_file}")
    logger.info("Fill in ratings (0-10 scale) for each aspect before training")


def main() -> None:
    """Main function for building preference pairs or creating annotation templates."""
    parser = argparse.ArgumentParser(
        description="Build preference pairs from ratings or create annotation templates"
    )
    
    parser.add_argument(
        "--ratings",
        type=str,
        help="Path to ratings.csv file (for building pairs from existing ratings)",
    )
    parser.add_argument(
        "--images-csv",
        type=str,
        help="Path to generated images CSV file (for creating annotation template)",
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["pairs", "multihead"],
        default="pairs",
        help="Output format: 'pairs' for preference pairs, "
             "'multihead' for rating template",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="data/prompts.csv",
        help="Path to prompts.csv file",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="data/images",
        help="Directory containing generated images",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for pairs.jsonl file or rating template",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum rating difference to create a pair",
    )
    
    args = parser.parse_args()
    
    output_file = Path(args.output)
    
    # Handle multihead annotation template creation
    if args.images_csv and args.format == "multihead":
        images_csv_file = Path(args.images_csv)
        if not images_csv_file.exists():
            raise FileNotFoundError(f"Images CSV file not found: {images_csv_file}")
        
        logger.info(f"Creating multi-head annotation template from {images_csv_file}")
        create_multihead_template(str(images_csv_file), output_file)
        logger.info("Multi-head annotation template created successfully")
        return
    
    # Handle existing ratings workflow (original functionality)
    if not args.ratings:
        raise ValueError("--ratings is required for building preference pairs")
    
    # Validate input files
    ratings_file = Path(args.ratings)
    prompts_file = Path(args.prompts)
    images_dir = Path(args.images)
    
    if not ratings_file.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_file}")
    
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Load data
    logger.info(f"Loading ratings from {ratings_file}")
    ratings_df = pd.read_csv(ratings_file)
    
    logger.info(f"Loading prompts from {prompts_file}")
    prompts_dict = load_prompts(str(prompts_file))
    
    # Validate ratings format
    required_columns = ["prompt_id", "seed", "rating"]
    if not all(col in ratings_df.columns for col in required_columns):
        raise ValueError(f"Ratings CSV must contain columns: {required_columns}")
    
    logger.info(f"Loaded {len(ratings_df)} ratings for {len(prompts_dict)} prompts")
    
    # Build preference pairs
    logger.info("Building preference pairs...")
    pairs = build_preference_pairs(
        ratings_df=ratings_df,
        prompts_dict=prompts_dict,
        images_dir=images_dir,
        threshold=args.threshold,
    )
    
    # Analyze pairs
    analyze_pairs(pairs)
    
    # Save pairs
    save_pairs(pairs, output_file)
    
    logger.info("Preference pairs generation completed successfully")


if __name__ == "__main__":
    main()
