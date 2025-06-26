# API Reference

## Main Classes

### DoRATrainer

The main training class for SDXL DoRA fine-tuning.

```python
from sdxl_dora_trainer import DoRATrainer, TrainingConfig

# Create configuration
config = TrainingConfig(
    dataset_path="./my_dataset",
    rank=64,
    alpha=32,
    learning_rate=5e-5
)

# Initialize trainer
trainer = DoRATrainer(config)

# Start training
trainer.train()
```

#### Methods

##### `__init__(config: TrainingConfig)`
Initialize the trainer with configuration.

**Parameters:**
- `config`: TrainingConfig object with training parameters

##### `train()`
Start the training process.

**Returns:**
- None

**Raises:**
- `RuntimeError`: If training fails
- `ValueError`: If configuration is invalid

##### `validate(step: int)`
Run validation and generate sample images.

**Parameters:**
- `step`: Current training step

##### `save_checkpoint(step: int)`
Save model checkpoint.

**Parameters:**
- `step`: Current training step

### TrainingConfig

Configuration dataclass for training parameters.

```python
@dataclass
class TrainingConfig:
    # Model and data paths
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    dataset_path: str = ""
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    
    # DoRA specific parameters
    rank: int = 32
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "to_k", "to_q", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2"
    ])
    
    # Training parameters
    learning_rate: float = 5e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 1000
    warmup_steps: int = 100
    save_every: int = 250
    eval_every: int = 100
    
    # Image parameters
    resolution: int = 1024
    center_crop: bool = True
    random_flip: bool = True
    
    # Mixed precision and memory
    mixed_precision: str = "no"  # fp16, bf16, no
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = True
```

### SDXLDoRAInference

Inference class for generating images with trained DoRA models.

```python
from inference import SDXLDoRAInference

# Initialize inference
inference = SDXLDoRAInference(
    base_model_path="stabilityai/stable-diffusion-xl-base-1.0",
    dora_weights_path="./output/checkpoints/checkpoint-1000",
    device="cuda"
)

# Load model
inference.load_model()

# Generate images
images = inference.generate_images(
    prompts=["a beautiful landscape"],
    num_inference_steps=50,
    guidance_scale=7.5
)
```

#### Methods

##### `__init__(base_model_path, dora_weights_path=None, device="auto")`
Initialize inference pipeline.

**Parameters:**
- `base_model_path`: Path to base SDXL model
- `dora_weights_path`: Path to DoRA weights (optional)
- `device`: Device to use ("auto", "cuda", "cpu", "mps")

##### `load_model()`
Load the model and DoRA weights.

##### `generate_images(prompts, negative_prompt="", num_images_per_prompt=1, num_inference_steps=50, guidance_scale=7.5, width=1024, height=1024, seed=None, output_dir="./generated_images")`
Generate images from prompts.

**Parameters:**
- `prompts`: List of text prompts
- `negative_prompt`: Negative prompt text
- `num_images_per_prompt`: Number of images per prompt
- `num_inference_steps`: Number of denoising steps
- `guidance_scale`: Classifier-free guidance scale
- `width`: Image width
- `height`: Image height
- `seed`: Random seed for reproducibility
- `output_dir`: Directory to save images

**Returns:**
- List of PIL Image objects

## Utility Functions

### Environment Checking

```python
from utils import check_environment, get_system_info

# Check if environment is properly configured
is_ready, issues = check_environment()

# Get detailed system information
system_info = get_system_info()
```

### Dataset Analysis

```python
from scripts.validate_dataset import analyze_dataset, get_dataset_stats

# Analyze dataset
stats = analyze_dataset("./my_dataset")

# Get basic statistics
image_count, caption_count, avg_size = get_dataset_stats("./my_dataset")
```

### Configuration Management

```python
from config_manager import ConfigManager

# Create default configuration
config_manager = ConfigManager()
config = config_manager.create_default_config()

# Save configuration
config_manager.save_config(config, "my_config.yaml")

# Load configuration
config = config_manager.load_config("my_config.yaml")

# Validate configuration
errors = config_manager.validate_config(config)
```

## Command Line Interface

### Main Training Script

```bash
python sdxl_dora_trainer.py [OPTIONS]
```

**Options:**
- `--dataset_path PATH`: Path to training dataset (required)
- `--config PATH`: Configuration file path
- `--model_name NAME`: Base model name or path
- `--output_dir PATH`: Output directory
- `--rank INT`: DoRA rank (default: 32)
- `--alpha INT`: DoRA alpha (default: 16)
- `--learning_rate FLOAT`: Learning rate (default: 5e-5)
- `--batch_size INT`: Batch size (default: 1)
- `--max_train_steps INT`: Maximum training steps (default: 1000)
- `--resolution INT`: Image resolution (default: 1024)
- `--mixed_precision CHOICE`: Mixed precision mode (default: no)
- `--report_to CHOICE`: Logging backend (tensorboard, wandb, none)

### Inference Script

```bash
python inference.py [OPTIONS]
```

**Options:**
- `--base_model PATH`: Base SDXL model path
- `--dora_weights PATH`: DoRA weights path
- `--prompt TEXT`: Text prompt (can be used multiple times)
- `--prompts_file PATH`: File with prompts (one per line)
- `--negative_prompt TEXT`: Negative prompt
- `--num_images INT`: Number of images per prompt
- `--steps INT`: Number of inference steps
- `--guidance_scale FLOAT`: Guidance scale
- `--width INT`: Image width
- `--height INT`: Image height
- `--seed INT`: Random seed
- `--output_dir PATH`: Output directory
- `--interactive`: Run in interactive mode

### Utility Scripts

#### Environment Check
```bash
python utils.py check-env
```

#### Dataset Validation
```bash
python scripts/validate_dataset.py --dataset PATH [--output PATH]
```

#### Debug DoRA Weights
```bash
python scripts/debug_dora_weights.py --weights_path PATH [--test_base]
```

#### Fix Corrupted Weights
```bash
python scripts/fix_dora_weights.py --weights_path PATH [--output_path PATH] [--clip_value FLOAT]
```

## Error Handling

### Common Exceptions

#### `TrainingConfigError`
Raised when training configuration is invalid.

```python
try:
    trainer = DoRATrainer(config)
except TrainingConfigError as e:
    print(f"Configuration error: {e}")
```

#### `ModelLoadError`
Raised when model loading fails.

```python
try:
    trainer.load_models()
except ModelLoadError as e:
    print(f"Model loading failed: {e}")
```

#### `DatasetError`
Raised when dataset is invalid or corrupted.

```python
try:
    trainer.create_dataset()
except DatasetError as e:
    print(f"Dataset error: {e}")
```

### Error Codes

- `E001`: Configuration validation failed
- `E002`: Model loading failed
- `E003`: Dataset creation failed
- `E004`: Training step failed
- `E005`: Checkpoint saving failed
- `E006`: Validation failed
- `E007`: Memory allocation failed

## Performance Optimization

### Memory Management

```python
# Enable gradient checkpointing
config.gradient_checkpointing = True

# Use 8-bit optimizer
config.use_8bit_adam = True

# Adjust batch size and accumulation
config.batch_size = 1
config.gradient_accumulation_steps = 8
```

### Speed Optimization

```python
# Use mixed precision (if stable)
config.mixed_precision = "bf16"

# Enable xformers
# pip install xformers

# Increase batch size if memory allows
config.batch_size = 2
```

### Multi-GPU Training

```bash
# Use accelerate launch
accelerate launch --multi_gpu --num_processes 2 sdxl_dora_trainer.py [OPTIONS]
```

## Logging and Monitoring

### TensorBoard Integration

```python
config.report_to = "tensorboard"
config.logging_dir = "./logs"
```

### Weights & Biases Integration

```python
config.report_to = "wandb"
config.project_name = "my-sdxl-project"
config.run_name = "experiment-1"
```

### Custom Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log custom metrics
logger.info(f"Training loss: {loss:.4f}")
```
