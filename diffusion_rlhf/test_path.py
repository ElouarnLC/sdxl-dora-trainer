#!/usr/bin/env python3
"""Test path resolution."""

from pathlib import Path

# Test path resolution
ratings_file = Path("data/annotations/multihead_ratings.csv")
print(f"Ratings file: {ratings_file}")
print(f"Ratings file exists: {ratings_file.exists()}")

image_path = "data/images/0/0.png"
image_path_resolved = Path(image_path)
print(f"Image path: {image_path_resolved}")
print(f"Image path exists: {image_path_resolved.exists()}")

# Check the actual files
import os
print(f"Current working directory: {os.getcwd()}")
print(f"Files in data/images/0/: {list(Path('data/images/0').glob('*')) if Path('data/images/0').exists() else 'Directory does not exist'}")
