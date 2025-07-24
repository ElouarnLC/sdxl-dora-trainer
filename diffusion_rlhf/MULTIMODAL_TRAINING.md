# Multimodal Reward Model Training

This directory contains an enhanced version of the reward model training that incorporates both **images and text prompts** for improved prompt-image alignment assessment. This is critical for evaluating prompt-conditioned generation models like SDXL.

## Overview

The multimodal reward model extends the original image-only approach by:

1. **Processing both images and prompts** using OpenCLIP's vision and text encoders
2. **Multiple fusion strategies** for combining visual and textual features
3. **Enhanced alignment assessment** for prompt-conditioned generation
4. **Specialized multi-head evaluation** considering prompt context

## Key Components

### 1. Multimodal Reward Model (`dspo/multimodal_reward.py`)

The `MultimodalMultiHeadReward` class extends the original model with:

- **Dual encoders**: Processes images and text using frozen OpenCLIP backbone
- **Feature fusion**: Three fusion methods for combining modalities
- **Specialized heads**: 5 MLP heads for different aspects with multimodal context

#### Fusion Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `concat` | Simple concatenation of features | Default, reliable approach |
| `add` | Element-wise addition | When features have same dimensionality |
| `cross_attention` | Attention-based fusion | Advanced prompt-image interaction |

### 2. Multimodal Datasets (`dspo/multimodal_datasets.py`)

Enhanced dataset classes that load both images and corresponding prompts:

- `MultimodalPairPreferenceDataset`: For preference pairs with prompt context
- `MultimodalMultiHeadPairPreferenceDataset`: For multi-aspect ratings with prompts

### 3. Training Script (`scripts/train_multimodal_reward.py`)

Complete training pipeline supporting both basic and multi-head modes with multimodal inputs.

## Quick Start

### Prerequisites

Ensure you have the required data files:
- `data/prompts.csv` - Prompt text mapping
- `data/annotations/multihead_ratings.csv` - Multi-head ratings (for multihead mode)
- Image pairs or ratings data

### Basic Training

```bash
# Basic multimodal training with preference pairs
python scripts/train_multimodal_reward.py \
    --mode pairs \
    --pairs data/pairs.jsonl \
    --prompts data/prompts_example.csv \
    --output outputs/multimodal_reward_basic \
    --epochs 10 \
    --batch-size 8 \
    --fusion-method concat
```

### Multi-head Training

```bash
# Multi-head training with different fusion methods
python scripts/train_multimodal_reward.py \
    --mode multihead \
    --ratings data/annotations/multihead_ratings.csv \
    --prompts data/prompts_example.csv \
    --output outputs/multimodal_reward_multihead \
    --epochs 15 \
    --batch-size 8 \
    --fusion-method cross_attention \
    --min-rating-diff 1.0
```

## Data Format

### Prompts File (`prompts.csv`)
```csv
prompt_id,prompt
0,"A serene mountain landscape at sunset"
1,"A futuristic city with flying cars"
```

### Ratings File (for multihead mode)
```csv
prompt_id,image_path,spatial_rating,icono_rating,style_rating,fidelity_rating,material_rating
0,data/images/0/1.png,8.5,7.2,9.1,8.8,7.5
0,data/images/0/2.png,6.1,8.3,7.4,9.2,8.1
```

## Architecture Details

### Model Architecture
```
Input: Images [B, 3, 224, 224] + Prompts [List[str]]
    ↓
OpenCLIP Encoders (frozen)
    ↓
Image Features [B, 768] + Text Features [B, 768]
    ↓
Feature Fusion (concat/add/cross_attention)
    ↓
Fused Features [B, fusion_dim]
    ↓
5 Specialized MLP Heads
    ↓
Head Outputs: spatial, icono, style, fidelity, material
    ↓
Weighted Combination → Final Score
```

### Specialized Heads

Each head evaluates a specific aspect with prompt context:

- **Spatial**: Composition and layout matching prompt descriptions
- **Icono**: Iconographic elements alignment with prompt content
- **Style**: Artistic style consistency with prompt style cues
- **Fidelity**: Image quality relative to prompt complexity
- **Material**: Material properties as described in prompts

## Training Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--mode` | Training mode | `pairs` | `pairs`, `multihead` |
| `--fusion-method` | Feature fusion strategy | `concat` | `concat`, `add`, `cross_attention` |
| `--prompts` | Path to prompts.csv | Required | - |
| `--epochs` | Number of epochs | 10 | - |
| `--batch-size` | Batch size | 8 | - |
| `--learning-rate` | Learning rate | 1e-4 | - |
| `--mixed-precision` | Precision mode | `no` | `no`, `fp16`, `bf16` |

## Benefits of Multimodal Training

### 1. Enhanced Prompt-Image Alignment
- Assesses how well images match their text descriptions
- Critical for prompt-conditioned generation evaluation
- Identifies semantic mismatches between prompts and images

### 2. Context-Aware Quality Assessment
- Evaluates image quality in context of prompt complexity
- Considers style requirements mentioned in prompts
- Assesses completeness of prompt fulfillment

### 3. Improved Generalization
- Better performance on diverse prompt-image pairs
- More robust to different artistic styles and content types
- Enhanced understanding of text-visual relationships

## Performance Monitoring

The training script logs comprehensive metrics:

```
Epoch 5 metrics:
  train_total_loss: 0.4521
  train_spatial_loss: 0.0892
  train_style_loss: 0.0756
  val_spatial_accuracy: 0.8234
  val_spatial_f1: 0.8156
  val_style_accuracy: 0.7891
  val_style_f1: 0.7823
```

Monitor these key metrics:
- **Per-head F1 scores**: Individual aspect performance
- **Combined accuracy**: Overall model performance
- **Head importance weights**: Learned aspect priorities

## Advanced Usage

### Custom Fusion Implementation

You can extend the fusion methods by modifying the `fuse_features` method in `MultimodalMultiHeadReward`:

```python
def fuse_features(self, image_features, text_features):
    # Custom fusion implementation
    return fused_features
```

### Evaluation Scripts

Use the trained model for inference:

```python
from dspo.multimodal_reward import MultimodalMultiHeadReward

# Load trained model
model = MultimodalMultiHeadReward.from_pretrained("outputs/multimodal_reward_best")

# Evaluate image-prompt pairs
images = load_images(["image1.jpg", "image2.jpg"])
prompts = ["A sunset over mountains", "A futuristic cityscape"]
scores = model(images, prompts)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use mixed precision
2. **Slow training**: Enable mixed precision (`--mixed-precision fp16`)
3. **Poor convergence**: Try different fusion methods or learning rates

### Memory Optimization

- Use `--mixed-precision bf16` for modern GPUs
- Reduce batch size if experiencing OOM errors
- Consider gradient accumulation for effective larger batch sizes

## Comparison with Image-Only Model

| Aspect | Image-Only | Multimodal |
|--------|------------|------------|
| Input | Images only | Images + Prompts |
| Context | Visual content | Visual + Semantic |
| Alignment | Style/quality only | Prompt-image alignment |
| Use Case | General quality | Prompt-conditioned generation |

The multimodal approach is particularly beneficial when training SDXL for specific prompts, as it directly optimizes for prompt-image alignment rather than just visual quality.
