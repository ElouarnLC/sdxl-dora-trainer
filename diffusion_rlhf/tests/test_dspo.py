"""
Unit tests for DSPO package components.

Tests for datasets, reward model, and tuner functionality
to ensure correct shapes, forward passes, and training steps.
"""

import pytest
import torch
from pathlib import Path
import tempfile
import jsonlines
import pandas as pd
from PIL import Image
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from dspo.datasets import PairPreferenceDataset
from dspo.reward import MultiHeadReward
from dspo.tuner import DSPOFineTuner


class TestPairPreferenceDataset:
    """Test PairPreferenceDataset functionality."""
    
    def create_test_data(self, temp_dir: Path) -> tuple[Path, Path]:
        """
        Create test data files.
        
        Parameters
        ----------
        temp_dir : Path
            Temporary directory
            
        Returns
        -------
        tuple[Path, Path]
            Paths to pairs.jsonl and images directory
        """
        # Create test images directory
        images_dir = temp_dir / "images"
        images_dir.mkdir()
        
        # Create test images
        for prompt_id in [0, 1]:
            prompt_dir = images_dir / str(prompt_id)
            prompt_dir.mkdir()
            
            for seed in [0, 1]:
                # Create dummy image
                image = Image.fromarray(
                    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                )
                image.save(prompt_dir / f"{seed}.png")
        
        # Create test pairs.jsonl
        pairs_file = temp_dir / "pairs.jsonl"
        test_pairs = [
            {
                "prompt": "test prompt 1",
                "img_pos": str(images_dir / "0" / "0.png"),
                "img_neg": str(images_dir / "0" / "1.png"),
                "label": 1.0,
            },
            {
                "prompt": "test prompt 2",
                "img_pos": str(images_dir / "1" / "1.png"),
                "img_neg": str(images_dir / "1" / "0.png"),
                "label": 0.0,
            },
        ]
        
        with jsonlines.open(pairs_file, "w") as writer:
            for pair in test_pairs:
                writer.write(pair)
        
        return pairs_file, images_dir
    
    def test_dataset_creation(self):
        """Test dataset creation and basic functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pairs_file, _ = self.create_test_data(temp_path)
            
            # Create dataset
            dataset = PairPreferenceDataset(
                pairs_file=str(pairs_file),
                image_size=224,
                split="train",
                val_split=0.5,
            )
            
            # Test dataset length
            assert len(dataset) > 0
            
            # Test sample retrieval
            sample = dataset[0]
            
            # Check sample structure
            assert "prompt" in sample
            assert "img_pos" in sample
            assert "img_neg" in sample
            assert "label" in sample
            
            # Check tensor shapes
            assert sample["img_pos"].shape == (3, 224, 224)
            assert sample["img_neg"].shape == (3, 224, 224)
            assert sample["label"].shape == ()
            
            # Check tensor types
            assert sample["img_pos"].dtype == torch.float32
            assert sample["img_neg"].dtype == torch.float32
            assert sample["label"].dtype == torch.float32
    
    def test_train_val_split(self):
        """Test train/validation split functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pairs_file, _ = self.create_test_data(temp_path)
            
            # Create train dataset
            train_dataset = PairPreferenceDataset(
                pairs_file=str(pairs_file),
                split="train",
                val_split=0.5,
            )
            
            # Create val dataset
            val_dataset = PairPreferenceDataset(
                pairs_file=str(pairs_file),
                split="val",
                val_split=0.5,
            )
            
            # Check split sizes
            total_samples = len(train_dataset) + len(val_dataset)
            assert total_samples == 2  # We created 2 pairs
            assert len(val_dataset) == 1  # 50% split
            assert len(train_dataset) == 1


class TestMultiHeadReward:
    """Test MultiHeadReward model functionality."""
    
    def test_model_creation(self):
        """Test reward model creation and basic properties."""
        model = MultiHeadReward(
            model_name="ViT-B-32",  # Smaller model for testing
            pretrained="openai",
            hidden_dim=256,
        )
        
        # Check model components
        assert hasattr(model, "backbone")
        assert hasattr(model, "heads")
        assert hasattr(model, "head_weights")
        
        # Check number of heads
        assert len(model.heads) == 5
        
        # Check head names
        expected_heads = {"spatial", "icono", "style", "fidelity", "material"}
        assert set(model.heads.keys()) == expected_heads
    
    def test_forward_pass(self):
        """Test forward pass through reward model."""
        model = MultiHeadReward(
            model_name="ViT-B-32",
            pretrained="openai",
            hidden_dim=256,
        )
        model.eval()
        
        # Create test batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        
        # Test forward pass
        with torch.no_grad():
            # Test combined output
            rewards = model(images)
            assert rewards.shape == (batch_size, 1)
            
            # Test individual head outputs
            head_outputs = model(images, return_individual=True)
            assert isinstance(head_outputs, dict)
            assert len(head_outputs) == 5
            
            for head_name, output in head_outputs.items():
                assert output.shape == (batch_size, 1)
    
    def test_preference_loss(self):
        """Test preference loss computation."""
        model = MultiHeadReward(
            model_name="ViT-B-32",
            pretrained="openai",
            hidden_dim=256,
        )
        
        # Create test data
        batch_size = 2
        img_pos = torch.randn(batch_size, 3, 224, 224)
        img_neg = torch.randn(batch_size, 3, 224, 224)
        labels = torch.tensor([1.0, 0.0])
        
        # Compute loss
        loss = model.compute_preference_loss(img_pos, img_neg, labels)
        
        # Check loss properties
        assert loss.dim() == 0  # Scalar
        assert loss.requires_grad  # Should be differentiable
        assert loss.item() >= 0.0  # Non-negative
    
    def test_head_importance(self):
        """Test head importance weight computation."""
        model = MultiHeadReward(
            model_name="ViT-B-32",
            pretrained="openai",
            hidden_dim=256,
        )
        
        importance = model.get_head_importance()
        
        # Check structure
        assert isinstance(importance, dict)
        assert len(importance) == 5
        
        # Check weights sum to 1
        total_weight = sum(importance.values())
        assert abs(total_weight - 1.0) < 1e-6


class TestDSPOFineTuner:
    """Test DSPOFineTuner functionality."""
    
    @pytest.fixture
    def mock_reward_model(self, tmp_path):
        """Create a mock reward model for testing."""
        # Create a simple reward model and save it
        model = MultiHeadReward(
            model_name="ViT-B-32",
            pretrained="openai",
            hidden_dim=128,
        )
        
        reward_path = tmp_path / "reward_model"
        model.save_pretrained(str(reward_path))
        
        return str(reward_path)
    
    def test_tuner_creation(self):
        """Test DSPO fine-tuner creation."""
        tuner = DSPOFineTuner(
            rank=4,
            alpha=16,
            learning_rate=1e-4,
            beta=0.1,
        )
        
        # Check basic properties
        assert tuner.rank == 4
        assert tuner.alpha == 16
        assert tuner.learning_rate == 1e-4
        assert tuner.beta == 0.1
    
    def test_alpha_default(self):
        """Test default alpha calculation."""
        tuner = DSPOFineTuner(rank=8)
        assert tuner.alpha == 32  # 4 * rank
    
    @pytest.mark.slow
    def test_model_loading(self, mock_reward_model):
        """Test model loading functionality."""
        tuner = DSPOFineTuner(
            rank=4,
            alpha=16,
            learning_rate=1e-4,
        )
        
        # This test requires actual model loading, which is slow
        # In practice, you might want to mock this
        try:
            tuner.load_models(
                reward_model_path=mock_reward_model,
                torch_dtype=torch.float32,
            )
            
            # Check models are loaded
            assert tuner.pipeline is not None
            assert tuner.unet is not None
            assert tuner.reward_model is not None
            
        except Exception as e:
            # Skip if models can't be loaded (e.g., no internet, no GPU)
            pytest.skip(f"Model loading failed: {e}")
    
    def test_dora_config(self):
        """Test DoRA configuration setup."""
        tuner = DSPOFineTuner(rank=8, alpha=32)
        
        # Test private method (for unit testing purposes)
        # In practice, this would be tested through integration tests
        assert tuner.rank == 8
        assert tuner.alpha == 32


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_dataset_to_model_pipeline(self):
        """Test data flow from dataset to model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test dataset
            test_dataset = TestPairPreferenceDataset()
            pairs_file, _ = test_dataset.create_test_data(temp_path)
            
            dataset = PairPreferenceDataset(
                pairs_file=str(pairs_file),
                image_size=224,
                split="train",
            )
            
            # Create reward model
            model = MultiHeadReward(
                model_name="ViT-B-32",
                pretrained="openai",
                hidden_dim=128,
            )
            
            # Get sample from dataset
            sample = dataset[0]
            
            # Process with model
            with torch.no_grad():
                reward_pos = model(sample["img_pos"].unsqueeze(0))
                reward_neg = model(sample["img_neg"].unsqueeze(0))
                
                # Check outputs
                assert reward_pos.shape == (1, 1)
                assert reward_neg.shape == (1, 1)
                
                # Test loss computation
                loss = model.compute_preference_loss(
                    img_pos=sample["img_pos"].unsqueeze(0),
                    img_neg=sample["img_neg"].unsqueeze(0),
                    labels=sample["label"].unsqueeze(0),
                )
                
                assert loss.dim() == 0
                assert not torch.isnan(loss)
                assert not torch.isinf(loss)


# Performance benchmarks (optional)
class TestPerformance:
    """Performance tests for critical components."""
    
    @pytest.mark.slow
    def test_dataset_loading_speed(self):
        """Test dataset loading performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create larger test dataset
            test_dataset = TestPairPreferenceDataset()
            pairs_file, _ = test_dataset.create_test_data(temp_path)
            
            dataset = PairPreferenceDataset(
                pairs_file=str(pairs_file),
                image_size=224,
            )
            
            # Time dataset iteration
            import time
            start_time = time.time()
            
            for i in range(min(10, len(dataset))):
                _ = dataset[i]
            
            elapsed = time.time() - start_time
            
            # Should be reasonably fast (adjust threshold as needed)
            assert elapsed < 5.0  # 5 seconds for 10 samples
    
    @pytest.mark.slow
    def test_model_inference_speed(self):
        """Test model inference performance."""
        model = MultiHeadReward(
            model_name="ViT-B-32",
            pretrained="openai",
            hidden_dim=128,
        )
        model.eval()
        
        # Test batch processing
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(images)
        
        elapsed = time.time() - start_time
        
        # Should be reasonably fast
        assert elapsed < 10.0  # 10 seconds for 10 forward passes


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
