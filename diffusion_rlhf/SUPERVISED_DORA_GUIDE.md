# DoRA Training on Vitraux Dataset with DSPO Evaluation

This document explains the correct training approach: supervised DoRA training on real vitraux images with DSPO-based evaluation.

## Overview

This approach combines the best of both worlds:

1. **Trains DoRA on real vitraux images** using supervised learning on image-text pairs
2. **Evaluates model iterations using DSPO methodology** - generates preference pairs and evaluates with reward model
3. **Selects the best model** based on DSPO preference scores

## Dataset Structure

The training uses your real vitraux dataset:

```
datasets/
└── vitraux/
    ├── item_42379_media_42380.jpg    # Real stained glass images
    ├── item_42379_media_42380.txt    # Detailed text descriptions
    ├── item_42381_media_42382.jpg
    ├── item_42381_media_42382.txt
    └── ...
```

## Training Process

### 1. Supervised Learning Phase

- **Input**: Real vitraux images + detailed text annotations
- **Target**: Train DoRA to reproduce the image-text pairs
- **Loss Function**: Standard diffusion loss (MSE between predicted and actual noise)
- **Objective**: Learn the vitraux style and iconography from real examples

### 2. DSPO Evaluation Phase

- **Generate preference pairs** using different CFG scales (7.5 vs 10.0)
- **Score pairs** using your trained reward model (F1=0.8333)
- **Compute DSPO metrics**: Preference scores, accuracy, DSPO loss
- **Save best model** based on DSPO preference scores

## Key Features

### Real Dataset Training
```python
class VitrauxDataset(Dataset):
    def __init__(self, dataset_path):
        # Find all image-text pairs
        self.image_files = list(Path(dataset_path).glob("*.jpg"))
        self.data_pairs = []
        
        for img_file in self.image_files:
            txt_file = img_file.with_suffix('.txt')
            if txt_file.exists():
                text = open(txt_file).read().strip()
                self.data_pairs.append({
                    'image_path': img_file,
                    'text': text
                })
```

### DSPO Evaluation
```python
def evaluate_with_dspo(checkpoint_path, reward_model_path, prompts):
    # Generate preference pairs with different CFG scales
    img_preferred = pipeline(prompt, guidance_scale=10.0, ...)
    img_dispreferred = pipeline(prompt, guidance_scale=7.5, ...)
    
    # Score with reward model
    comparison = reward_model.compare_images(img_preferred, img_dispreferred, prompt)
    
    # Compute DSPO metrics
    preference_diff = reward_preferred - reward_dispreferred
    dspo_loss = -log(sigmoid(beta * preference_diff))
```

### Progressive Model Selection
- Saves checkpoint every 500 steps
- Evaluates every 500 steps using DSPO methodology
- Keeps track of best performing model based on DSPO preference scores
- Saves evaluation metrics to CSV

## Usage

### Command Line
```bash
python scripts/train_dora_vitraux.py \
    --dataset-path datasets/vitraux \
    --reward-model outputs/full_enhanced_final/best/model.pt \
    --output-dir outputs/dora_vitraux_training \
    --batch-size 2 \
    --num-steps 2000 \
    --rank 16 \
    --lr 5e-5 \
    --eval-every 500 \
    --save-every 500
```

### Using Scripts
```bash
# Linux/Mac
./run_supervised_dora_training.sh

# Windows PowerShell
./run_supervised_dora_training.ps1
```

## Output Structure

```
outputs/dora_vitraux_training/
├── checkpoint-500/              # Regular checkpoints
├── checkpoint-1000/
├── checkpoint-1500/
├── final_model/                 # Final checkpoint
├── best_model/                  # Best performing model (highest DSPO score)
└── dspo_evaluation_results.csv  # DSPO evaluation metrics over time
```

## DSPO Evaluation Metrics

The `dspo_evaluation_results.csv` contains:
- `step`: Training step
- `loss`: Training loss
- `dspo_preference_score`: Average preference score from reward model
- `dspo_accuracy`: Accuracy on preference pairs (higher CFG should be better)
- `dspo_loss`: DSPO loss computed from preference scores
- `num_evaluated`: Number of pairs evaluated

## Example Text Annotations

From your vitraux dataset:
```
isabel stuart kneeling in gothic niche, royal robes, red mantle with lion rampant, 
crowned noblewoman praying, tiled floor, brocade curtain, gothic architecture, 
prayer book on yellow-covered lectern, stained glass style, medieval devotional scene, 
scottish heraldry, late 15th century, sacred interior
```

## Advantages

1. **Uses real vitraux images** instead of synthetic data
2. **Rich text annotations** describing style, iconography, and historical context
3. **DSPO-based evaluation** provides objective quality assessment
4. **Progressive model selection** finds the best performing checkpoint
5. **Maintains training history** for detailed analysis

## Monitoring Training

Watch for:
- **Decreasing training loss**: Supervised learning is working
- **Increasing DSPO preference scores**: Model quality is improving according to reward model
- **High DSPO accuracy**: Model consistently generates better images with higher CFG
- **Stable DSPO loss**: Training convergence

## Next Steps

After training completes:

1. **Analyze DSPO results**: Check `dspo_evaluation_results.csv` for trends
2. **Use best model**: Load from `best_model/` directory
3. **Generate vitraux samples**: Test the model on new prompts
4. **Compare checkpoints**: Evaluate different iterations

This approach ensures you're training on authentic vitraux data while using advanced DSPO methodology for objective model evaluation and selection.
