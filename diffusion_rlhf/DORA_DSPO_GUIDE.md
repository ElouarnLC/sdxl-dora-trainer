# Enhanced DSPO Training with DoRA - Complete Workflow Guide

## 🎯 What You Have Accomplished

You've successfully achieved **F1=0.8333** with your enhanced multimodal reward model (a 27% improvement over the 0.65-0.66 baseline)! Now you're ready to use this high-performance model for DSPO training with DoRA fine-tuning.

## 🏗️ Architecture Overview

### Your Enhanced Reward Model (F1=0.8333)
- **Backbone**: ViT-L-14-336 (high-resolution 336px input)
- **Architecture**: Multi-head reward model with 5 specialized heads
- **Training**: Enhanced with attention fusion, focal loss, EMA, and MixUp
- **Performance**: F1=0.8333 (best-in-class for your dataset)

### DoRA Fine-Tuning
- **Method**: Weight-Decomposed Low-Rank Adaptation (superior to LoRA)
- **Target**: SDXL UNet layers for efficient adaptation
- **Benefits**: Better performance than LoRA with similar efficiency

## 🚀 Quick Start

### 1. Check Your Setup
```bash
python diffusion_rlhf/scripts/quick_start_dora_dspo.py
```

This will verify:
- ✅ Your trained reward model exists (`outputs/full_enhanced_final/best/model.pt`)
- ✅ DoRA DSPO training script is ready
- ✅ Example prompts are available

### 2. Start DoRA DSPO Training
```bash
python diffusion_rlhf/scripts/train_dspo_with_dora.py \
  --prompts diffusion_rlhf/data/prompts_example.csv \
  --output outputs/dspo_dora_training \
  --num-pairs 100 \
  --epochs 10 \
  --lr 1e-5
```

## 📋 Command Line Options

| Parameter | Description | Default | Recommendation |
|-----------|-------------|---------|----------------|
| `--prompts` | Path to prompts CSV file | Required | Use your best prompts |
| `--output` | Output directory | `outputs/dspo_dora` | Descriptive name |
| `--reward-model` | Path to reward model | `outputs/full_enhanced_final/best/model.pt` | Your trained model |
| `--model-name` | SDXL model name | `stabilityai/stable-diffusion-xl-base-1.0` | Standard SDXL |
| `--num-pairs` | Preference pairs to generate | 200 | Start with 50-100 |
| `--epochs` | Training epochs | 10 | 5-15 typical range |
| `--lr` | Learning rate | 1e-5 | Conservative start |

## 🎛️ DoRA Configuration

The script uses optimized DoRA settings:

```python
# DoRA Configuration
rank = 32           # Low-rank dimension
alpha = 32.0        # Scaling factor
dropout = 0.1       # Regularization
target_modules = [  # SDXL UNet layers
    "to_k", "to_q", "to_v", "to_out.0",  # Attention
    "ff.net.0.proj", "ff.net.2",         # Feed-forward
    "proj_in", "proj_out",                # Projection
]
```

## 📊 Training Process

### Phase 1: Preference Pair Generation
1. **Image Generation**: Create images with different sampling parameters
2. **Reward Scoring**: Use your F1=0.8333 model to evaluate quality
3. **Pair Creation**: Build preference dataset based on reward scores

### Phase 2: DSPO Training
1. **DoRA Application**: Apply Weight-Decomposed Low-Rank Adaptation
2. **Preference Learning**: Train using Direct Statistical Preference Optimization
3. **Checkpoint Saving**: Regular saves with best model selection

### Phase 3: Evaluation & Deployment
1. **Quality Assessment**: Generate test images and compare
2. **Model Export**: Save DoRA weights for inference
3. **Integration**: Use fine-tuned model in production

## 📈 Expected Results

### Performance Improvements
- **Image Quality**: Higher fidelity and detail
- **Prompt Alignment**: Better adherence to text prompts
- **Consistency**: More stable generation across different seeds
- **Efficiency**: DoRA provides good performance with fewer parameters

### Training Metrics
- **Reward Margin**: Preference difference between chosen/rejected images
- **Loss Convergence**: DSPO loss should decrease steadily
- **Quality Scores**: Your reward model will track improvements

## 🔧 Advanced Configuration

### Custom Prompts
```csv
prompt,weight
"A serene mountain landscape at sunset",1.0
"Portrait of a cat wearing sunglasses",1.2
"Abstract art with vibrant colors",0.8
```

### Memory Optimization
For limited GPU memory:
```bash
python diffusion_rlhf/scripts/train_dspo_with_dora.py \
  --num-pairs 25 \
  --epochs 5 \
  --lr 5e-6
```

### High-Performance Training
For powerful hardware:
```bash
python diffusion_rlhf/scripts/train_dspo_with_dora.py \
  --num-pairs 500 \
  --epochs 20 \
  --lr 2e-5
```

## 🎯 Key Features

### 1. DoRA vs LoRA
- **DoRA**: Decomposes weight updates into magnitude and direction
- **Benefits**: Better fine-tuning quality with similar efficiency
- **Usage**: Automatically applied to SDXL UNet layers

### 2. High-Performance Reward Model Integration
- **F1=0.8333**: Your trained model provides accurate quality assessment
- **Multi-Head Analysis**: 5 specialized heads for comprehensive evaluation
- **Real-Time Scoring**: Efficient preference pair generation

### 3. Optimized DSPO Training
- **Stable Learning**: Conservative learning rates for reliable convergence
- **Memory Efficient**: Gradient accumulation and checkpointing
- **Robust Training**: Built-in error handling and recovery

## 📁 Output Structure

After training, you'll have:

```
outputs/dspo_dora_training/
├── dora_weights/              # DoRA fine-tuned weights
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── training_state.pt          # Training checkpoints
├── preference_pairs/          # Generated training data
└── logs/                      # Training logs and metrics
```

## 🚀 Next Steps

### 1. Inference with Fine-Tuned Model
```python
from diffusers import StableDiffusionXLPipeline
from peft import PeftModel

# Load base model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0"
)

# Load DoRA weights
pipe.unet = PeftModel.from_pretrained(
    pipe.unet, 
    "outputs/dspo_dora_training/dora_weights"
)

# Generate improved images
image = pipe("A beautiful landscape").images[0]
```

### 2. Batch Processing
```bash
python scripts/batch_generate_dora.py \
  --model outputs/dspo_dora_training \
  --prompts batch_prompts.csv \
  --output improved_images/
```

### 3. Model Distribution
- Share DoRA weights (lightweight, ~100MB)
- Users can apply to any SDXL model
- Maintains compatibility with existing workflows

## 🎉 Success Metrics

You'll know the training is successful when:
- ✅ **Reward scores increase** over training epochs
- ✅ **Generated images show better quality** compared to base SDXL
- ✅ **Prompt adherence improves** based on your reward model
- ✅ **Training converges** without overfitting

## 🆘 Troubleshooting

### Common Issues
1. **CUDA OOM**: Reduce `--num-pairs` or add `--gradient-accumulation-steps 8`
2. **Slow training**: Start with `--num-pairs 25` for testing
3. **Poor convergence**: Try `--lr 5e-6` for more stable training

### Support Files
- `diffusion_rlhf/scripts/train_dspo_with_dora.py` - Main training script
- `diffusion_rlhf/scripts/quick_start_dora_dspo.py` - Setup checker
- `dspo/trained_reward_model.py` - Your F1=0.8333 model wrapper

Ready to create state-of-the-art SDXL fine-tuning with your high-performance reward model! 🎨✨
