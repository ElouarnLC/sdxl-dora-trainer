# Multi-Head Reward Model Training

## Overview

The multi-head reward model has been enhanced to support fine-grained preference learning across 5 specialized aspects:

1. **Spatial**: Composition, layout, rule of thirds, balance, leading lines
2. **Iconographic**: Symbols, cultural elements, narrative content, object recognition
3. **Style**: Artistic technique, color harmony, brushwork, aesthetic movement
4. **Fidelity**: Technical quality, sharpness, noise, artifacts, resolution
5. **Material**: Texture realism, surface properties, lighting interaction

## Data Format

### Option 1: Multi-Head Ratings (Recommended)

For detailed stylistic assessment, use the multi-head ratings format:

```csv
prompt_id,image_path,spatial_rating,icono_rating,style_rating,fidelity_rating,material_rating
1,data/images/1/image1.png,8.2,7.5,6.8,9.1,7.3
1,data/images/1/image2.png,6.1,5.9,5.2,7.8,6.2
2,data/images/2/image3.png,9.0,8.8,7.9,8.5,8.1
```

**Rating Scale**: 0-10 for each aspect
- 0-3: Poor quality
- 4-6: Average quality  
- 7-8: Good quality
- 9-10: Excellent quality

### Option 2: Binary Pairs (Legacy)

For overall preference comparisons:

```jsonl
{"prompt": "A serene landscape", "img_pos": "path/to/better.png", "img_neg": "path/to/worse.png", "label": 1.0}
```

## Training Modes

### Multi-Head Training (Recommended)

```bash
python train_reward.py \
    --mode multihead \
    --ratings data/annotations/multihead_ratings.csv \
    --output outputs/multihead_reward_model \
    --epochs 15 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --min-rating-diff 0.5
```

**Benefits**:
- Individual head specialization
- Interpretable per-aspect scores
- Better fine-grained preference learning
- Detailed performance metrics

### Binary Pairs Training (Legacy)

```bash
python train_reward.py \
    --mode pairs \
    --pairs data/annotations/pairs.jsonl \
    --output outputs/pairs_reward_model \
    --epochs 10 \
    --batch-size 8
```

## Model Architecture

The model consists of:

1. **Frozen OpenCLIP ViT-L/14**: Feature extraction backbone
2. **5 Specialized MLP Heads**: One per aspect
3. **Learnable Combination Weights**: For final score aggregation

```python
# Individual head outputs
head_outputs = model(images, return_individual=True)
spatial_score = head_outputs["spatial"]
style_score = head_outputs["style"]
# ... etc

# Combined score
combined_score = model(images)  # Weighted combination
```

## Performance Metrics

### Multi-Head Training Metrics
- Per-head accuracy and F1 score
- Combined accuracy and F1 score
- Individual head loss tracking
- Head importance weights

### Legacy Training Metrics
- Overall accuracy and F1 score
- Binary preference loss

## Example Usage

```python
from dspo.reward import MultiHeadReward

# Load trained model
model = MultiHeadReward.from_pretrained("outputs/multihead_reward_model/best")

# Get individual aspect scores
head_scores = model(images, return_individual=True)
print(f"Spatial score: {head_scores['spatial']}")
print(f"Style score: {head_scores['style']}")

# Get combined score
combined = model(images)
print(f"Combined score: {combined}")

# Check head importance
importance = model.get_head_importance()
print(f"Head weights: {importance}")
```

## Key Advantages

1. **Granular Assessment**: Each aspect evaluated independently
2. **Interpretability**: Understand which aspects drive preferences
3. **Debugging**: Identify underperforming aspects
4. **Customization**: Adjust head weights for specific use cases
5. **Better Training**: More targeted learning signals

## Migration Guide

To migrate from the current single-rating system:

1. **Expand your ratings.csv** to include all 5 aspects
2. **Use the multihead training mode** for better performance
3. **Analyze per-head metrics** to understand model behavior
4. **Adjust training parameters** based on individual head performance

The multi-head approach provides significantly better fine-grained stylistic understanding and is recommended for all new training runs.
