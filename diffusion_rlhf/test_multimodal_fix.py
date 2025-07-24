#!/usr/bin/env python3
"""
Test script to verify the multimodal reward model initialization fix.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from dspo.multimodal_reward import MultimodalMultiHeadReward

def test_multimodal_reward_init():
    """Test multimodal reward model initialization."""
    print("Testing multimodal reward model initialization...")
    
    try:
        # Test with concat fusion (default)
        model = MultimodalMultiHeadReward(fusion_method="concat")
        print("✅ Concat fusion method: SUCCESS")
        
        # Test with add fusion
        model = MultimodalMultiHeadReward(fusion_method="add")
        print("✅ Add fusion method: SUCCESS")
        
        # Test with cross_attention fusion
        model = MultimodalMultiHeadReward(fusion_method="cross_attention")
        print("✅ Cross-attention fusion method: SUCCESS")
        
        print("\n🎉 All multimodal reward model initialization tests passed!")
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_multimodal_reward_init()
    sys.exit(0 if success else 1)
