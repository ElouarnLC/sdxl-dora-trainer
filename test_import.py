#!/usr/bin/env python3
"""Test script to verify dspo import works."""

import sys
from pathlib import Path

# Add the diffusion_rlhf directory to Python path
script_dir = Path(__file__).parent
repo_root = script_dir
diffusion_rlhf_path = repo_root / "diffusion_rlhf"
sys.path.append(str(diffusion_rlhf_path))

print(f"Added to path: {diffusion_rlhf_path}")
print(f"Path exists: {diffusion_rlhf_path.exists()}")

try:
    from dspo.tuner import DSPOFineTuner
    print("✅ Successfully imported DSPOFineTuner")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Python path:")
    for p in sys.path:
        print(f"  {p}")
