#!/usr/bin/env python3
"""
Test script to check imports and provide diagnostics
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all required imports."""
    print("🔍 Testing imports...")
    
    # Add the diffusion_rlhf directory to Python path
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    diffusion_rlhf_path = repo_root / "diffusion_rlhf"
    sys.path.append(str(diffusion_rlhf_path))
    
    print(f"📁 Added path: {diffusion_rlhf_path}")
    print(f"📁 Path exists: {diffusion_rlhf_path.exists()}")
    
    # Test basic imports
    try:
        import torch
        print("✅ torch imported successfully")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        from torch.utils.data import Dataset, DataLoader
        print("✅ torch.utils.data imported successfully")
    except ImportError as e:
        print(f"❌ torch.utils.data import failed: {e}")
        return False
    
    # Test DSPO imports step by step
    try:
        print("🧪 Testing DSPO imports...")
        
        # Test if we can find the dspo package
        import dspo
        print("✅ dspo package imported successfully")
        
        # Test specific modules
        from dspo import reward
        print("✅ dspo.reward imported successfully")
        
        from dspo.reward import MultiHeadReward
        print("✅ MultiHeadReward imported successfully")
        
        from dspo.tuner import DSPOFineTuner
        print("✅ DSPOFineTuner imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ DSPO import failed: {e}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        
        # Try to identify which specific dependency is missing
        print("\n🔍 Checking specific dependencies...")
        
        deps = [
            "open_clip",
            "peft", 
            "accelerate",
            "diffusers",
            "transformers",
            "safetensors"
        ]
        
        for dep in deps:
            try:
                __import__(dep)
                print(f"✅ {dep} available")
            except ImportError:
                print(f"❌ {dep} MISSING")
        
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 All imports successful!")
    else:
        print("\n❌ Some imports failed. Please install missing dependencies.")
