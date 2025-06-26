# Installation Guide

## System Requirements

- **Operating System**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **GPU**: NVIDIA GPU with 12GB+ VRAM (16GB+ recommended)
- **CUDA**: 11.8 or 12.1
- **Storage**: 20GB+ free disk space
- **RAM**: 16GB+ system RAM recommended

## Supported GPUs

### Recommended (16GB+ VRAM)
- RTX 4090 (24GB)
- RTX 4080 (16GB)
- RTX 3090 (24GB)
- RTX 3090 Ti (24GB)
- A100 (40GB/80GB)
- H100 (80GB)

### Minimum (12GB+ VRAM)
- RTX 4070 Ti (12GB)
- RTX 3080 Ti (12GB)
- RTX 3080 (10GB/12GB)
- RTX Titan (24GB)
- Quadro RTX 6000 (24GB)

## Installation Methods

### Method 1: Automated Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/sdxl-dora-trainer.git
cd sdxl-dora-trainer

# Run automated setup
python setup.py

# Verify installation
python utils.py check-env
```

### Method 2: Manual Installation

#### Step 1: Install PyTorch

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only (not recommended):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Step 2: Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

#### Step 3: Create Directories

```bash
mkdir -p output cache logs datasets
```

### Method 3: Conda Environment

```bash
# Create conda environment
conda create -n sdxl-dora python=3.10
conda activate sdxl-dora

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Method 4: Docker (Advanced)

```bash
# Build Docker image
docker build -t sdxl-dora-trainer .

# Run with GPU support
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output sdxl-dora-trainer
```

## Verification

After installation, verify everything works:

```bash
# Check environment
python utils.py check-env

# Test basic functionality
python tests/test_setup.py

# Test GPU allocation
python tests/test_device_allocation.py
```

## Common Installation Issues

### CUDA Version Mismatch
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Install matching PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Permission Errors (Linux/macOS)
```bash
# Use user installation
pip install --user -r requirements.txt

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

### Memory Issues
```bash
# Clear pip cache
pip cache purge

# Install with no cache
pip install --no-cache-dir -r requirements.txt
```

### Windows-Specific Issues

**Long Path Support:**
```powershell
# Enable long path support (run as Administrator)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**Visual C++ Build Tools:**
Download and install Microsoft C++ Build Tools if you get compilation errors.

## Environment Variables

Set these environment variables for optimal performance:

```bash
# Windows (PowerShell)
$env:CUDA_VISIBLE_DEVICES="0"
$env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Linux/macOS
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Updating

To update to the latest version:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstallation

```bash
# Remove conda environment
conda remove -n sdxl-dora --all

# Or remove pip packages
pip uninstall -r requirements.txt -y

# Remove project directory
rm -rf sdxl-dora-trainer
```
