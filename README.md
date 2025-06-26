# SDXL DoRA Trainer

A production-ready tool for fine-tuning Stable Diffusion XL models using **DoRA (Weight-Decomposed Low-Rank Adaptation)**, which is more efficient and effective than traditional LoRA.

## 🚨 Important: Black Image Issue Fixed!

**If you're experiencing black/empty images during generation, see [docs/BLACK_IMAGE_FIX.md](docs/BLACK_IMAGE_FIX.md) for the solution!**

**TL;DR:** Use `--mixed_precision no` instead of `fp16` to fix black image generation with DoRA.

## 🚀 Features

- **DoRA Integration**: Uses the latest DoRA technique for more efficient fine-tuning
- **Production Ready**: Robust error handling, logging, and monitoring
- **Memory Optimized**: Support for gradient checkpointing, 8-bit optimizers, and mixed precision
- **Flexible Configuration**: YAML/JSON config files and command-line arguments
- **Multi-GPU Support**: Built on Accelerate for distributed training
- **Comprehensive Logging**: TensorBoard and Weights & Biases integration
- **CLI Tools**: Utilities for dataset analysis and environment checking
- **Cross-Platform**: Works on Windows, Linux, and macOS

## 📋 Requirements

- Python 3.8+
- NVIDIA GPU with 12GB+ VRAM (16GB+ recommended for 1024px training)
- CUDA 11.8 or 12.1
- 20GB+ free disk space

## 🛠️ Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/ElouarnLC/sdxl-dora-trainer.git
cd sdxl-dora-trainer

# Run the automated setup
python setup.py
```

### Option 2: Manual Setup

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Create directories
mkdir -p output cache logs datasets
```

## 🚦 Quick Start

### 1. Prepare Your Dataset

Organize your training images in a folder structure like this:

```
my_dataset/
├── image1.jpg
├── image1.txt  (caption file)
├── image2.png
├── image2.txt
└── ...
```

**Caption files are optional** - if not provided, the filename will be used as the caption.

### 2. Check Your Environment

```bash
python utils.py check-env
```

### 3. Analyze Your Dataset

```bash
python scripts/validate_dataset.py --dataset ./my_dataset
```

### 4. Get Training Suggestions

```bash
python utils.py suggest-params --dataset ./my_dataset
```

### 5. Start Training

#### Basic Training
```bash
python sdxl_dora_trainer.py --dataset_path ./my_dataset --output_dir ./output
```

#### Advanced Training with Custom Parameters
```bash
python sdxl_dora_trainer.py \
  --dataset_path ./my_dataset \
  --output_dir ./output \
  --rank 128 \
  --alpha 64 \
  --learning_rate 5e-5 \
  --batch_size 2 \  --max_train_steps 2000 \
  --resolution 1024 \
  --mixed_precision no \
  --report_to wandb \
  --project_name my-sdxl-project
```

#### Using Configuration File
```bash
# Edit config.yaml with your settings
python sdxl_dora_trainer.py --config config.yaml
```

## ⚙️ Configuration

### Key Parameters

| Parameter | Description | Default | Recommendations |
|-----------|-------------|---------|-----------------|
| `dataset_path` | Path to training images | Required | Folder with images and captions |
| `rank` | DoRA rank (affects model size) | 64 | 32-128 based on dataset size |
| `alpha` | DoRA alpha parameter | 32 | Usually rank/2 |
| `learning_rate` | Learning rate | 1e-4 | 5e-5 to 2e-4 |
| `batch_size` | Batch size | 1 | Increase based on GPU memory |
| `max_train_steps` | Training steps | 1000 | 500-3000 based on dataset |
| `resolution` | Image resolution | 1024 | 512, 768, or 1024 |
| `mixed_precision` | Precision mode | no | no for stability, bf16 for speed |

### Full Configuration Example

```yaml
# Dataset and Model
dataset_path: "./my_images"
model_name: "stabilityai/stable-diffusion-xl-base-1.0"
output_dir: "./output"

# DoRA Parameters
rank: 128
alpha: 64
dropout: 0.1

# Training
learning_rate: 5e-5
batch_size: 2
gradient_accumulation_steps: 2
max_train_steps: 2000
resolution: 1024

# Memory Optimization
mixed_precision: "no"  # Use "no" for stability, "bf16" for speed
gradient_checkpointing: true
use_8bit_adam: true

# Logging
report_to: "wandb"
project_name: "my-custom-model"
```

## 🔧 Utilities

### Environment Check
```bash
python utils.py check-env
```
Validates your environment, checks GPU memory, and lists system information.

### Dataset Analysis
```bash
python scripts/validate_dataset.py --dataset ./my_images
```
Analyzes your dataset and provides statistics on images, captions, and sizes.

### Parameter Suggestions
```bash
python utils.py suggest-params --dataset ./my_images --gpu-memory 24
```
Suggests optimal training parameters based on your dataset and hardware.

### Configuration Management
```bash
# Create default config
python config_manager.py create --config config.yaml

# Validate config
python config_manager.py validate --config config.yaml
```

## 📊 Monitoring Training

### TensorBoard (Default)
```bash
tensorboard --logdir ./logs
```

### Weights & Biases
Set `report_to: "wandb"` in your config or use `--report_to wandb`.

### Training Outputs

During training, the tool will:
- Save checkpoints every 250 steps (configurable)
- Generate validation images every 100 steps
- Log training metrics continuously
- Display progress with rich terminal interface

## 📁 Output Structure

After training, your output directory will look like this:

```
output/
├── checkpoints/                    # Model checkpoints
│   ├── checkpoint-250/
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── training_args.bin
│   ├── checkpoint-500/
│   └── ...
├── samples/                        # Validation images
│   ├── step_100/
│   ├── step_200/
│   └── ...
└── final_model/                    # Final trained model
    ├── adapter_config.json
    └── adapter_model.safetensors

logs/
├── tensorboard_logs/               # TensorBoard logs
└── training.log                    # Text logs
```

## 🎯 DoRA vs LoRA

DoRA (Weight-Decomposed Low-Rank Adaptation) offers several advantages over traditional LoRA:

- **Better Performance**: Higher quality results with the same parameter count
- **More Stable Training**: Less prone to overfitting
- **Improved Convergence**: Faster and more reliable training
- **Better Fine-grained Control**: More precise adaptation to your dataset

## 💡 Tips for Better Results

### Dataset Preparation
- Use high-quality images (avoid blurry, low-res, or corrupted images)
- Write descriptive captions that match your target style
- Aim for 50-500 images for most use cases
- Keep aspect ratios consistent when possible

### Training Parameters
- Start with lower rank (32-64) for smaller datasets
- Use higher learning rates (1e-4 to 2e-4) for better convergence
- Increase batch size if you have sufficient GPU memory
- Use longer training for complex datasets

### Memory Optimization
- Enable gradient checkpointing for 40% memory savings
- Use 8-bit Adam optimizer to reduce memory usage
- Use fp16 mixed precision for 2x speedup
- Reduce batch size and increase gradient accumulation steps if OOM

## 🐛 Troubleshooting

### Common Issues

**Out of Memory (OOM)**
```bash
# Reduce batch size
--batch_size 1 --gradient_accumulation_steps 8

# Enable memory optimizations
--gradient_checkpointing --use_8bit_adam --mixed_precision no

# Use lower resolution
--resolution 768
```

**Black Images During Generation**
See [docs/BLACK_IMAGE_FIX.md](docs/BLACK_IMAGE_FIX.md) for the complete solution.
```bash
# Quick fix: use mixed_precision no
--mixed_precision no
```

**Training Too Slow**
```bash
# Use mixed precision (if stable)
--mixed_precision bf16

# Increase batch size if possible
--batch_size 2

# Use xformers for attention optimization
pip install xformers
```

**Poor Results**
```bash
# Increase training steps
--max_train_steps 2000

# Adjust learning rate
--learning_rate 5e-5

# Use higher rank
--rank 128 --alpha 64
```

**Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check environment
python utils.py check-env
```

## 📁 Repository Structure

```
sdxl-dora-trainer/
├── 📄 README.md                    # Main documentation
├── 📄 LICENSE                      # MIT License
├── 📄 CHANGELOG.md                 # Version history
├── 📄 pyproject.toml               # Modern Python packaging
├── 📄 setup.cfg                    # Setup configuration
├── 📄 requirements.txt             # Production dependencies
├── 📄 requirements-dev.txt         # Development dependencies
├── 📄 .gitignore                   # Git ignore rules
│
├── 📄 sdxl_dora_trainer.py         # Main training script
├── 📄 inference.py                 # Inference script
├── 📄 config_manager.py            # Configuration management
├── 📄 utils.py                     # Utility functions
├── 📄 setup.py                     # Automated setup script
│
├── 📂 docs/                        # Documentation
│   ├── 📄 BLACK_IMAGE_FIX.md       # ⚠️ IMPORTANT: Black image fix
│   ├── 📄 INSTALLATION.md          # Installation guide
│   ├── 📄 API.md                   # API reference
│   └── 📄 CONTRIBUTING.md          # Contributing guidelines
│
├── 📂 examples/                    # Example configurations
│   ├── 📄 basic_config.yaml        # Basic training config
│   ├── 📄 advanced_config.yaml     # Advanced training config
│   ├── 📄 memory_optimized_config.yaml # Low-memory config
│   ├── 📄 train_example.py         # Training example script
│   └── 📄 inference_example.py     # Inference example script
│
├── 📂 scripts/                     # Utility scripts
│   ├── 📄 validate_dataset.py      # Dataset validation
│   ├── 📄 debug_dora_weights.py    # Debug DoRA weights
│   ├── 📄 fix_dora_weights.py      # Fix corrupted weights
│   └── 📄 batch_train.py           # Batch training script
│
├── 📂 tests/                       # Test suite
│   ├── 📄 test_setup.py            # Setup tests
│   ├── 📄 test_device_allocation.py # GPU tests
│   ├── 📄 test_sdxl_simple.py      # SDXL tests
│   └── 📄 test_vae_encoding.py     # VAE tests
│
└── 📂 .github/                     # GitHub specific files
    ├── 📂 ISSUE_TEMPLATE/
    │   ├── 📄 bug_report.md
    │   └── 📄 feature_request.md
    └── 📄 pull_request_template.md
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
git clone https://github.com/ElouarnLC/sdxl-dora-trainer.git
cd sdxl-dora-trainer
pip install -r requirements.txt
pip install -e .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Diffusers](https://github.com/huggingface/diffusers) - The foundation for stable diffusion training
- [PEFT](https://github.com/huggingface/peft) - DoRA implementation
- [Accelerate](https://github.com/huggingface/accelerate) - Distributed training support
- [Rich](https://github.com/Textualize/rich) - Beautiful terminal interface

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{sdxl_dora_trainer,
  title={SDXL DoRA Trainer: Production-Ready Fine-tuning Tool},
  author={ElouarnLC},
  year={2025},
  url={https://github.com/ElouarnLC/sdxl-dora-trainer},
  license={MIT}
}
```

---

**Happy Training! 🎨✨**
