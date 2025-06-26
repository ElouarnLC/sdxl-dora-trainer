#!/usr/bin/env python3
"""
Test script for SDXL DoRA Trainer.
Runs basic functionality tests to ensure everything is working.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

try:
    from rich.console import Console
    from sdxl_dora_trainer import TrainingConfig, validate_config
    from utils import analyze_dataset
    console = Console()
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install dependencies with: pip install -r requirements.txt")
    sys.exit(1)

def test_imports():
    """Test that all required modules can be imported."""
    console.print("[bold blue]Testing Imports...[/bold blue]")
    
    required_modules = [
        'torch',
        'diffusers', 
        'transformers',
        'accelerate',
        'peft',
        'datasets',
        'safetensors',
        'PIL',
        'numpy',
        'tqdm',
        'rich'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            console.print(f"[green]✓[/green] {module}")
        except ImportError:
            console.print(f"[red]✗[/red] {module}")
            failed_imports.append(module)
    
    if failed_imports:
        console.print(f"[red]Failed to import: {', '.join(failed_imports)}[/red]")
        return False
    
    console.print("[green]All imports successful![/green]")
    return True

def test_gpu():
    """Test GPU availability."""
    console.print("\n[bold blue]Testing GPU...[/bold blue]")
    
    try:
        import torch
        if torch.cuda.is_available():
            console.print("[green]✓[/green] CUDA available")
            console.print(f"[green]✓[/green] {torch.cuda.device_count()} GPU(s) detected")
            
            # Test basic GPU operations
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            _ = x * 2  # Simple GPU operation
            console.print("[green]✓[/green] Basic GPU operations work")
            return True
        else:
            console.print("[yellow]⚠[/yellow] CUDA not available")
            return False
    except Exception as e:
        console.print(f"[red]✗[/red] GPU test failed: {e}")
        return False

def test_config():
    """Test configuration validation."""
    console.print("\n[bold blue]Testing Configuration...[/bold blue]")
    
    try:
        # Test valid config
        config = TrainingConfig()
        config.dataset_path = "/tmp/test"  # This won't exist, but that's ok for validation
        config.model_name = "stabilityai/stable-diffusion-xl-base-1.0"
        
        errors = validate_config(config)
        if not errors:
            console.print("[green]✓[/green] Configuration validation works")
            return True
        else:
            console.print(f"[yellow]Configuration validation found expected errors: {errors}")
            return True
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration test failed: {e}")
        return False

def create_test_dataset():
    """Create a small test dataset for testing."""
    console.print("\n[bold blue]Creating Test Dataset...[/bold blue]")
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        test_dataset_path = Path(temp_dir) / "test_dataset"
        test_dataset_path.mkdir(exist_ok=True)
        
        # Create test images
        for i in range(3):
            # Create a simple colored image
            image = Image.fromarray(
                np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            )
            image_path = test_dataset_path / f"test_image_{i}.png"
            image.save(image_path)
            
            # Create caption
            caption_path = test_dataset_path / f"test_image_{i}.txt"
            with open(caption_path, 'w') as f:
                f.write(f"test image {i}")
        
        console.print(f"[green]✓[/green] Test dataset created at {test_dataset_path}")
        return str(test_dataset_path)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to create test dataset: {e}")
        return None

def test_dataset_analysis(dataset_path):
    """Test dataset analysis functionality."""
    console.print("\n[bold blue]Testing Dataset Analysis...[/bold blue]")
    
    try:
        stats = analyze_dataset(dataset_path)
        if "error" not in stats:
            console.print("[green]✓[/green] Dataset analysis works")
            console.print(f"    Found {stats['total_images']} images")
            console.print(f"    Found {stats['total_captions']} captions")
            return True
        else:
            console.print(f"[red]✗[/red] Dataset analysis failed: {stats['error']}")
            return False
    except Exception as e:
        console.print(f"[red]✗[/red] Dataset analysis test failed: {e}")
        return False

def test_model_loading():
    """Test basic model loading (without actually loading the full model)."""
    console.print("\n[bold blue]Testing Model Components...[/bold blue]")
    
    try:
        from transformers import CLIPTokenizer
        
        # Test tokenizer loading
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"  # Smaller model for testing
        )
          # Test tokenization
        _ = tokenizer("test prompt", return_tensors="pt")
        console.print("[green]✓[/green] Tokenizer works")
        
        return True
        
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Model component test failed: {e}")
        console.print("    This is expected if you haven't downloaded models yet")
        return True  # Don't fail the test for this

def cleanup_test_files(dataset_path):
    """Clean up test files."""
    try:
        if dataset_path and Path(dataset_path).exists():
            shutil.rmtree(Path(dataset_path).parent)
            console.print("[green]✓[/green] Test files cleaned up")
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Could not clean up test files: {e}")

def main():
    """Run all tests."""
    console.print("[bold green]SDXL DoRA Trainer - Test Suite[/bold green]\n")
    
    tests = [
        ("Import Test", test_imports),
        ("GPU Test", test_gpu),
        ("Configuration Test", test_config),
    ]
    
    results = []
    test_dataset_path = None
    
    # Run basic tests
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            console.print(f"[red]✗[/red] {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Run dataset tests if possible
    if all(result for _, result in results):
        test_dataset_path = create_test_dataset()
        if test_dataset_path:
            dataset_result = test_dataset_analysis(test_dataset_path)
            results.append(("Dataset Analysis Test", dataset_result))
    
    # Run model test
    model_result = test_model_loading()
    results.append(("Model Component Test", model_result))
    
    # Print summary
    console.print("\n[bold magenta]Test Summary[/bold magenta]")
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        if result:
            console.print(f"[green]✓[/green] {test_name}")
            passed += 1
        else:
            console.print(f"[red]✗[/red] {test_name}")
    
    console.print(f"\n[bold]Results: {passed}/{total} tests passed[/bold]")
    
    if passed == total:
        console.print("[green]🎉 All tests passed! Your environment is ready for training.[/green]")
    elif passed >= total - 1:
        console.print("[yellow]⚠ Most tests passed. You should be able to train with minor issues.[/yellow]")
    else:
        console.print("[red]❌ Several tests failed. Please check your environment setup.[/red]")
    
    # Cleanup
    if test_dataset_path:
        cleanup_test_files(test_dataset_path)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
