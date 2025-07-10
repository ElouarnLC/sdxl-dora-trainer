# DSPO + SDXL: Diffusion Reinforcement Learning from Human Feedback

## Overview

This subproject implements **DSPO (Diffusion Statistical Preference Optimization)** for fine-tuning SDXL models using human feedback. It combines multi-head reward modeling with DoRA (Weight-Decomposed Low-Rank Adaptation) for efficient and effective model improvement.

## Architecture

- **Multi-head Reward Model**: OpenCLIP ViT-L/14 backbone with 5 specialized heads (spatial, iconographic, style, fidelity, material)
- **DSPO Fine-tuning**: DoRA-based fine-tuning using preference pairs
- **Optuna Optimization**: Automated hyperparameter search for optimal training configuration

## Quick Start

### 1. Generate Training Images
```bash
python scripts/generate_images.py --prompts data/prompts.csv --output data/images/
```

### 2. Annotate Images (Manual Step)
Rate generated images and create `data/annotations/ratings.csv`

### 3. Build Preference Pairs
```bash
python scripts/build_pairs.py --ratings data/annotations/ratings.csv --output data/annotations/pairs.jsonl
```

### 4. Train Reward Model
```bash
python scripts/train_reward.py --pairs data/annotations/pairs.jsonl --output outputs/reward_model/
```

### 5. Fine-tune with DSPO
```bash
python scripts/train_dspo.py --reward outputs/reward_model/best.pt --pairs data/annotations/pairs.jsonl
```

### 6. Hyperparameter Optimization
```bash
python scripts/sweep_dspo_optuna.py --study-name dspo-sweep --n-trials 20
```

## Data Format

### prompts.csv
```csv
prompt_id,prompt
1,"A serene mountain landscape at sunset"
2,"A futuristic cityscape with flying cars"
```

### ratings.csv
```csv
prompt_id,image_path,rating
1,data/images/1/uuid1.png,8.5
1,data/images/1/uuid2.png,6.2
```

### pairs.jsonl
```jsonl
{"prompt": "A serene mountain...", "img_pos": "path1.png", "img_neg": "path2.png", "label": 1.0}
```

## Requirements

- Python 3.11+
- NVIDIA GPU with 24GB+ VRAM (H100 recommended)
- CUDA 12.1+

