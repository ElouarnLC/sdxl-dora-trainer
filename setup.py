#!/usr/bin/env python3
"""
Setup script for SDXL DoRA Trainer.
Handles environment setup, dependency installation, and initial configuration.
"""

import sys
import subprocess
import platform
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

def run_command(command, description="Running command", check=True):
    """Run a shell command with progress indication."""
    console.print(f"[cyan]{description}...[/cyan]")
    
    try:
        if platform.system() == "Windows":
            # Use PowerShell on Windows
            result = subprocess.run(
                ["powershell.exe", "-Command", command],
                capture_output=True,
                text=True,
                check=check
            )
        else:
            # Use bash on Unix-like systems
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=check
            )
        
        if result.returncode == 0:
            console.print(f"[green]✓[/green] {description} completed successfully")
            return result.stdout
        else:
            console.print(f"[red]✗[/red] {description} failed")
            if result.stderr:
                console.print(f"[red]Error: {result.stderr}[/red]")
            return None
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗[/red] {description} failed: {e}")
        return None
    except FileNotFoundError:
        console.print(f"[red]✗[/red] Command not found: {command}")
        return None

def check_python_version():
    """Check if Python version is compatible."""
    console.print(Panel("[bold blue]Checking Python Version[/bold blue]"))
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        console.print(f"[red]✗[/red] Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    console.print(f"[green]✓[/green] Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_gpu():
    """Check GPU availability."""
    console.print(Panel("[bold blue]Checking GPU[/bold blue]"))
    
    try:
        if platform.system() == "Windows":
            result = run_command("nvidia-smi", "Checking NVIDIA GPU", check=False)
        else:
            result = run_command("nvidia-smi", "Checking NVIDIA GPU", check=False)
        
        if result:
            console.print("[green]✓[/green] NVIDIA GPU detected")
            return True
        else:
            console.print("[yellow]⚠[/yellow] No NVIDIA GPU detected or nvidia-smi not available")
            console.print("[yellow]  Training will be very slow on CPU[/yellow]")
            return False
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Could not check GPU: {e}")
        return False

def install_dependencies():
    """Install Python dependencies."""
    console.print(Panel("[bold blue]Installing Dependencies[/bold blue]"))
    
    # Check if pip is available
    pip_cmd = "python -m pip" if platform.system() == "Windows" else "pip3"
    
    # Upgrade pip first
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    
    # Install PyTorch first (with CUDA support if available)
    console.print("[cyan]Installing PyTorch with CUDA support...[/cyan]")
    torch_install_cmd = f"{pip_cmd} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    run_command(torch_install_cmd, "Installing PyTorch")
    
    # Install other dependencies
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        run_command(f"{pip_cmd} install -r {requirements_file}", "Installing other dependencies")
    else:
        console.print("[yellow]⚠[/yellow] requirements.txt not found, installing core packages individually")
        
        core_packages = [
            "diffusers>=0.24.0",
            "transformers>=4.25.0", 
            "accelerate>=0.25.0",
            "peft>=0.7.0",
            "datasets>=2.14.0",
            "safetensors>=0.3.0",
            "bitsandbytes",
            "xformers",
            "rich>=13.0.0",
            "pillow>=9.0.0",
            "numpy>=1.21.0",
            "tqdm>=4.64.0",
            "pyyaml>=6.0.0"
        ]
        
        for package in core_packages:
            run_command(f"{pip_cmd} install {package}", f"Installing {package}")

def create_directories():
    """Create necessary directories."""
    console.print(Panel("[bold blue]Creating Directories[/bold blue]"))
    
    directories = [
        "output",
        "cache", 
        "logs",
        "datasets",
        "output/checkpoints",
        "output/samples"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created {directory}/")

def create_example_config():
    """Create an example configuration if it doesn't exist."""
    console.print(Panel("[bold blue]Setting up Configuration[/bold blue]"))
    
    config_path = Path("config.yaml")
    if not config_path.exists():
        console.print("[yellow]Creating example configuration file...[/yellow]")
        
        example_config = """# SDXL DoRA Training Configuration
# Edit this file with your specific settings

# REQUIRED: Set your dataset path
dataset_path: "./datasets/your_images"  # Change this to your image folder

# Model settings
model_name: "stabilityai/stable-diffusion-xl-base-1.0"
output_dir: "./output"
cache_dir: "./cache"

# DoRA parameters (adjust based on your needs)
rank: 64
alpha: 32
dropout: 0.1

# Training parameters
learning_rate: 0.0001
batch_size: 1  # Increase if you have more GPU memory
gradient_accumulation_steps: 4
max_train_steps: 1000
resolution: 1024

# Memory optimization
mixed_precision: "fp16"
gradient_checkpointing: true
use_8bit_adam: true

# Logging
report_to: "tensorboard"  # Change to "wandb" if you prefer Weights & Biases
project_name: "my-sdxl-dora-training"
"""
        
        with open(config_path, "w") as f:
            f.write(example_config)
        
        console.print(f"[green]✓[/green] Created {config_path}")
        console.print("[yellow]  Please edit config.yaml with your dataset path and preferences[/yellow]")
    else:
        console.print(f"[green]✓[/green] Configuration file exists: {config_path}")

def test_installation():
    """Test if the installation works."""
    console.print(Panel("[bold blue]Testing Installation[/bold blue]"))
    
    # Test imports
    test_imports = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("accelerate", "Accelerate"),
        ("rich", "Rich")
    ]
    
    all_good = True
    for module, name in test_imports:
        try:
            __import__(module)
            console.print(f"[green]✓[/green] {name} import successful")
        except ImportError as e:
            console.print(f"[red]✗[/red] {name} import failed: {e}")
            all_good = False
    
    # Test CUDA if available
    try:
        import torch
        if torch.cuda.is_available():
            console.print(f"[green]✓[/green] CUDA available with {torch.cuda.device_count()} GPU(s)")
            console.print(f"    GPU 0: {torch.cuda.get_device_name(0)}")
        else:
            console.print("[yellow]⚠[/yellow] CUDA not available")
    except Exception as e:
        console.print(f"[red]✗[/red] Could not test CUDA: {e}")
        all_good = False
    
    return all_good

def main():
    """Main setup function."""
    console.print(Panel.fit(
        "[bold green]SDXL DoRA Trainer Setup[/bold green]\n"
        "This script will set up your environment for SDXL fine-tuning with DoRA.",
        title="Setup",
        border_style="green"
    ))
    
    # Check Python version
    if not check_python_version():
        console.print("[red]Please install Python 3.8 or higher and try again.[/red]")
        sys.exit(1)
    
    # Check GPU
    has_gpu = check_gpu()
    if not has_gpu:
        response = input("\nNo GPU detected. Training will be very slow. Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            console.print("Setup cancelled.")
            sys.exit(0)
    
    # Install dependencies
    try:
        install_dependencies()
    except KeyboardInterrupt:
        console.print("\n[yellow]Installation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Installation failed: {e}[/red]")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create example config
    create_example_config()
    
    # Test installation
    if test_installation():
        console.print(Panel(
            "[bold green]Setup Complete![/bold green]\n\n"
            "Next steps:\n"
            "1. Edit config.yaml with your dataset path\n"
            "2. Prepare your training images in a folder\n"
            "3. Run: python sdxl_dora_trainer.py --dataset_path your_images\n\n"
            "For help: python sdxl_dora_trainer.py --help\n"
            "For utilities: python utils.py check-env",
            title="Success",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[red]Setup completed with some issues.[/red]\n\n"
            "Please check the error messages above and resolve them before training.",
            title="Warning",
            border_style="yellow"
        ))

if __name__ == "__main__":
    main()
