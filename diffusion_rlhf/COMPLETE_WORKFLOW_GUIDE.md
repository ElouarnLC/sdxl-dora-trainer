# Complete Multi-Head RLHF Workflow Guide for SDXL

This guide explains the complete **Multi-Head Reinforcement Learning from Human Feedback (RLHF)** pipeline for improving SDXL image generation using specialized aesthetic evaluation.

## 🎯 What is Multi-Head RLHF for Image Generation?

Multi-Head RLHF trains AI models using **specialized reward heads** that evaluate different aesthetic aspects independently. Instead of simple preference comparisons, experts rate images across 5 specialized dimensions, providing fine-grained feedback for targeted improvements.

**Key Innovation**: Rather than asking "which image is better?", we ask experts to rate specific aesthetic aspects like composition, style, and technical quality separately. This enables precise optimization of each aesthetic dimension.

## 🧠 The 5 Specialized Reward Heads

1. **Spatial Head**: Composition, layout, rule of thirds, visual balance
2. **Iconographic Head**: Symbols, cultural elements, narrative content
3. **Style Head**: Artistic technique, color harmony, aesthetic movement
4. **Fidelity Head**: Technical quality, sharpness, noise, artifacts  
5. **Material Head**: Texture realism, surface properties, lighting

## 🔄 Complete Workflow Overview

```
Text Prompts → Generate Images → Multi-Aspect Expert Ratings → Train 5-Head Reward Model → DSPO Optimization → Enhanced SDXL
```

## 📋 Step-by-Step Process

### Step 1: Image Generation 🖼️
**Command**: `make generate-images-dual`

**What happens**:
- Takes text prompts from `data/prompts.csv`
- Generates 2 images per prompt using different seeds (0 and 1)
- Uses dual GPU setup for ~1.8x speed improvement
- Saves images to `data/images/` with metadata

**Output**: 
- Images: `data/images/prompt_id/seed.png`
- Metadata: `generated_dual_gpu.csv`

**Why 2 images per prompt?**: Provides variation for comprehensive expert evaluation across all aesthetic dimensions.

### Step 2: Create Multi-Head Rating Template 📝
**Command**: `make annotate`

**What happens**:
- Creates structured rating templates for expert evaluation
- Each image gets individual assessment across 5 aesthetic aspects
- Generates CSV template for systematic annotation

**Output**: `data/annotations/multihead_ratings.csv` template
```csv
prompt_id,image_path,spatial_rating,icono_rating,style_rating,fidelity_rating,material_rating
# Ready for expert annotation
```

### Step 3: Multi-Head Expert Annotation 👥🎯
**Manual Step** - Specialized aesthetic evaluation!

**What experts do**:
Each expert evaluates **every image individually** across all 5 aesthetic dimensions using a 0-10 scale:

#### 🎨 **Spatial Assessment (0-10)**:
- Composition and layout quality
- Rule of thirds application
- Visual balance and weight distribution
- Focal point effectiveness

#### 🏛️ **Iconographic Assessment (0-10)**:
- Symbolic content accuracy
- Cultural element representation
- Narrative coherence
- Conceptual depth

#### 🖌️ **Style Assessment (0-10)**:
- Artistic technique execution
- Color harmony and palette
- Aesthetic movement alignment
- Visual coherence

#### 🔍 **Fidelity Assessment (0-10)**:
- Technical image quality
- Sharpness and detail clarity
- Absence of artifacts/noise
- Overall visual fidelity

#### 🌟 **Material Assessment (0-10)**:
- Texture realism and detail
- Surface property accuracy
- Lighting interaction quality
- Physical believability

**Expert creates**: `data/annotations/multihead_ratings.csv`
```csv
prompt_id,image_path,spatial_rating,icono_rating,style_rating,fidelity_rating,material_rating
42,data/images/42/0.png,8,7,9,8,9  # High-quality sunset image
42,data/images/42/1.png,6,8,7,7,8  # Good but less balanced composition
43,data/images/43/0.png,9,6,8,9,7  # Excellent composition, moderate iconography
```

**Important Quality Standards**: 
- Multiple experts rate the same images for reliability
- Each rating represents absolute quality (0=poor, 10=excellent)
- Minimum 500+ rated images per aesthetic dimension
- Inter-annotator agreement >0.7 for reliable training

### Step 4: Train Multi-Head Reward Model 🏆🧠
**Command**: `make train-reward`

**What happens**:
- Trains a **5-head neural network** to predict expert ratings across all aesthetic dimensions
- **Shared backbone**: Frozen OpenCLIP ViT-L/14 for feature extraction
- **Specialized heads**: Each head learns one aesthetic aspect independently
- **Combination weights**: Learns optimal weighting between aesthetic aspects

**Technical Architecture**:
```
Image + Prompt → OpenCLIP ViT-L/14 → Shared Features
                                    ├── Spatial Head → Rating (0-10)
                                    ├── Iconographic Head → Rating (0-10) 
                                    ├── Style Head → Rating (0-10)
                                    ├── Fidelity Head → Rating (0-10)
                                    └── Material Head → Rating (0-10)
```

**Training Process**:
- Each head optimized independently on corresponding ratings
- Mean Squared Error loss per head
- Validation on held-out expert ratings
- Early stopping based on combined head performance

**Output**: `outputs/multihead_reward_model/`
- Individual head weights and metrics
- Combined scoring system
- Performance breakdown per aesthetic dimension

### Step 5: Multi-Head DSPO Training 🎯🚀
**Command**: `make train-dspo`

**What happens**:
- **DSPO** = Direct Statistical Preference Optimization with **Multi-Head Feedback**
- Fine-tunes SDXL using specialized aesthetic guidance from all 5 reward heads
- Each head provides targeted optimization signals for its aesthetic dimension
- Model learns to excel across **all aesthetic aspects simultaneously**

**Multi-Head Training Process**:
1. Generate images with current SDXL model
2. **Multi-dimensional scoring**: All 5 heads evaluate each image
3. **Weighted optimization**: Update SDXL to improve across all aesthetic dimensions
4. **Balanced improvement**: Prevent over-optimization of single aspects

**Technical Innovation**:
- **Composite loss**: Combines signals from all 5 specialized heads
- **Aspect weighting**: Dynamically balances aesthetic dimensions
- **Specialized gradients**: Each head contributes targeted improvements
- **Holistic optimization**: Results in images excellent across all criteria

**Output**: `outputs/dspo_model/` - Your **multi-dimensionally optimized** SDXL model!

**Key Advantage**: Unlike single-metric optimization, produces images that excel in composition AND style AND technical quality AND material realism AND iconographic accuracy.

## 🔧 Multi-GPU Setup Details

### Why Dual GPUs?
- **Single GPU**: ~2.5 images/minute
- **Dual GPU**: ~4.5 images/minute (1.8x speedup)
- **Quad GPU**: ~9 images/minute (3.6x speedup)

### How It Works (`generate_dual_gpu.py`):
1. **Two separate pipelines**: Each GPU runs a complete SDXL pipeline
2. **Work distribution**: Tasks distributed via queue system
3. **Memory isolation**: Each GPU has its own memory space
4. **No communication overhead**: Unlike DataParallel, no inter-GPU sync needed

### Memory Requirements:
- **Per GPU**: 12-18 GB VRAM for full SDXL pipeline
- **DoRA weights**: +1-2 GB additional
- **Generation buffer**: +2-3 GB

## 🎭 Role of Multi-Head Expert Annotators

### What Expert Annotators Actually Do:

#### Individual Image Assessment Protocol:
1. **View each image independently** (no pairwise comparison)
2. **Read the text prompt** carefully for context
3. **Rate across all 5 aesthetic dimensions** using 0-10 scale
4. **Apply specialized expertise** for each aesthetic aspect

### Detailed Rating Guidelines:

#### 🎨 **Spatial Rating (0-10)**:
```
Prompt: "A serene lake with mountains in the background"

Rating 9: Perfect rule of thirds, mountain placement creates depth, 
         lake reflection adds balance, strong focal hierarchy

Rating 5: Centered composition, adequate but static, 
         no clear focal point, acceptable but unremarkable

Rating 2: Poor composition, elements competing for attention,
         unbalanced weight distribution, confusing layout
```

#### 🏛️ **Iconographic Rating (0-10)**:
```
Prompt: "Medieval knight in shining armor"

Rating 9: Authentic heraldic symbols, period-accurate armor details,
         strong narrative presence, rich symbolic content

Rating 5: Generic knight imagery, basic symbolic elements,
         adequate medieval reference, limited depth

Rating 2: Anachronistic elements, poor symbolic accuracy,
         weak narrative coherence, cultural inconsistency
```

#### 🖌️ **Style Rating (0-10)**:
```
Prompt: "Impressionist painting of a garden"

Rating 9: Authentic brushwork technique, perfect color harmony,
         captures impressionist light quality, cohesive aesthetic

Rating 5: Some impressionist elements, decent color choices,
         adequate artistic technique, recognizable style

Rating 2: Poor artistic execution, clashing colors,
         inconsistent technique, style mismatch
```

#### 🔍 **Fidelity Rating (0-10)**:
```
Any image assessment:

Rating 9: Crystal clear details, no artifacts, perfect sharpness,
         excellent technical execution throughout

Rating 5: Generally clear, minor artifacts, acceptable quality,
         some soft areas but overall decent

Rating 2: Significant artifacts, blurry details, poor quality,
         distracting technical issues
```

#### 🌟 **Material Rating (0-10)**:
```
Prompt: "Wooden table with metal bowl"

Rating 9: Wood grain perfectly rendered, metal reflects realistically,
         authentic surface properties, believable materials

Rating 5: Recognizable materials, adequate surface properties,
         basic realism, acceptable texture quality

Rating 2: Poor material representation, unrealistic surfaces,
         flat textures, unconvincing properties
```

### Quality Control for Multi-Head Annotation:
- **Multiple annotators**: 3-5 experts per image for reliability across all dimensions
- **Specialized expertise**: Art critics for Style, photographers for Fidelity, etc.
- **Inter-annotator agreement**: Measure consistency per aesthetic dimension (target >0.7)
- **Calibration sessions**: Training workshops for consistent rating standards
- **Difficult cases**: Flag images with high annotator disagreement
- **Cross-validation**: Ensure head-specific expertise doesn't introduce bias

## 📊 Multi-Head Model Architecture During Training

### Multi-Head Reward Model Training:
```python
# Simplified multi-head training loop
for batch in training_data:
    image, prompt, spatial_rating, icono_rating, style_rating, fidelity_rating, material_rating = batch
    
    # Shared feature extraction
    features = clip_backbone(image, prompt)
    
    # Individual head predictions
    spatial_pred = spatial_head(features)
    icono_pred = iconographic_head(features)
    style_pred = style_head(features)
    fidelity_pred = fidelity_head(features)
    material_pred = material_head(features)
    
    # Individual losses per aesthetic dimension
    spatial_loss = mse_loss(spatial_pred, spatial_rating)
    icono_loss = mse_loss(icono_pred, icono_rating)
    style_loss = mse_loss(style_pred, style_rating)
    fidelity_loss = mse_loss(fidelity_pred, fidelity_rating)
    material_loss = mse_loss(material_pred, material_rating)
    
    # Combined optimization
    total_loss = spatial_loss + icono_loss + style_loss + fidelity_loss + material_loss
    optimizer.step()
```

### Multi-Head DSPO Training:
```python
# Simplified multi-head DSPO training
for prompt in training_prompts:
    # Generate multiple candidate images
    images = sdxl_model.generate(prompt, num_samples=4)
    
    # Multi-dimensional scoring from all 5 heads
    spatial_scores = [spatial_head(img, prompt) for img in images]
    icono_scores = [iconographic_head(img, prompt) for img in images]
    style_scores = [style_head(img, prompt) for img in images]
    fidelity_scores = [fidelity_head(img, prompt) for img in images]
    material_scores = [material_head(img, prompt) for img in images]
    
    # Weighted composite score (learnable weights)
    composite_scores = [
        w1*spatial + w2*icono + w3*style + w4*fidelity + w5*material
        for spatial, icono, style, fidelity, material in 
        zip(spatial_scores, icono_scores, style_scores, fidelity_scores, material_scores)
    ]
    
    # Update SDXL to optimize across all aesthetic dimensions
    loss = dspo_loss(images, composite_scores)
    sdxl_optimizer.step()
```

## 🚀 Getting Started with Multi-Head RLHF

### Quick Start (2 GPUs):
```bash
# 1. Generate images (takes ~30 min for 100 prompts)
make generate-images-dual

# 2. Create multi-head annotation template
make annotate

# 3. Have experts rate images across 5 aesthetic dimensions (manual step)
# Fill out data/annotations/multihead_ratings.csv with 0-10 ratings per aspect

# 4. Train 5-head reward model (takes ~2 hours)
make train-reward

# 5. Train multi-head optimized SDXL (takes ~4 hours)
make train-dspo

# 6. Test your aesthetically-enhanced model!
python inference.py --model outputs/dspo_model --prompt "Your test prompt"
```

### Single GPU Fallback:
```bash
make generate-images-single  # Slower but works for smaller datasets
```

### Test Pipeline:
```bash
make test-pipeline  # Quick test with small dataset to verify workflow
```

### Monitor Multi-Head Training:
```bash
make monitor-training  # Watch training progress across all aesthetic heads
```

## 📈 Expected Results from Multi-Head RLHF

### Performance Improvements:
- **Multi-dimensional accuracy**: 70-90% correlation with expert ratings per aesthetic head
- **Balanced quality**: Simultaneous improvement across all 5 aesthetic dimensions
- **Specialized optimization**: Targeted enhancement of composition, style, fidelity, materials, iconography
- **Holistic excellence**: Images score highly across multiple aesthetic criteria
- **Reduced trade-offs**: Minimizes improvement in one area at expense of others

### Aesthetic Dimension Improvements:
- **Spatial**: Better composition, rule of thirds, visual balance
- **Iconographic**: Richer symbolic content, cultural accuracy, narrative depth  
- **Style**: Improved artistic technique, color harmony, aesthetic coherence
- **Fidelity**: Higher technical quality, reduced artifacts, sharper details
- **Material**: More realistic textures, believable surface properties

### Timeline:
- **Image generation**: 2-8 hours (depending on dataset size)
- **Multi-head annotation**: 2-5 days (5x ratings per image, specialized expertise needed)
- **Multi-head reward training**: 2-3 hours (5 heads + combination weights)
- **Multi-head DSPO training**: 4-8 hours (multi-dimensional optimization)
- **Total**: 4-10 days for complete multi-head pipeline

## 🎯 Success Metrics for Multi-Head System

### Quantitative (Per Aesthetic Head):
- **Individual head accuracy**: >75% correlation with expert ratings per dimension
- **Composite score improvement**: Higher combined aesthetic ratings vs base model
- **Balanced improvement**: Gains across all 5 dimensions (no single-aspect optimization)
- **Expert preference win rate**: >65% vs base model across all aesthetic criteria

### Qualitative Assessment:
- **Multi-dimensional expert evaluation**: Professional assessment across all 5 aspects
- **Blind comparison studies**: Users prefer multi-head optimized images
- **Aesthetic portfolio review**: Art professionals evaluate holistic quality
- **Specialized domain testing**: Genre-specific evaluation (portraits, landscapes, etc.)

## 💡 Pro Tips for Multi-Head RLHF Success

### For Better Multi-Head Results:
1. **Diverse prompt portfolio**: Include various styles, subjects, complexity levels across all aesthetic dimensions
2. **Specialized expert panels**: Use domain experts (artists for Style, photographers for Fidelity, etc.)
3. **Consistent rating standards**: Detailed guidelines and calibration sessions per aesthetic dimension
4. **Balanced dataset**: Ensure variety in each aesthetic aspect to prevent head bias
5. **Iterative multi-head refinement**: Multiple rounds focusing on weakest aesthetic dimensions
6. **Cross-validation**: Test model performance on unseen prompts across all 5 aspects

### Multi-Head Specific Considerations:
- **Head interdependence**: Monitor for conflicts between aesthetic dimensions
- **Weighting optimization**: Tune combination weights based on target application
- **Annotation fatigue**: Rotate experts across aesthetic dimensions to maintain quality
- **Dimension balance**: Ensure no single aesthetic head dominates the optimization

### Common Multi-Head Pitfalls:
- **Insufficient dimension coverage**: Need diverse examples for each aesthetic aspect
- **Expert specialization bias**: Single expert rating all dimensions introduces perspective bias
- **Head competition**: Aesthetic dimensions optimizing against each other
- **Overfitting to rating style**: Model learns annotator quirks rather than true aesthetics
- **Imbalanced training**: Unequal quality/quantity of ratings across aesthetic dimensions

### Multi-Head Troubleshooting:
- **Poor individual head performance**: Need more specialized training data for that dimension
- **Conflicting aesthetic optimization**: Adjust combination weights or add regularization
- **Single-dimension overfitting**: Increase penalties for neglecting other aesthetic aspects
- **Training instability**: Lower learning rates, add gradient clipping for multi-head optimization
- **Memory issues with 5 heads**: Use gradient checkpointing or reduced batch sizes

This multi-head pipeline transforms your SDXL model from generating "technically adequate" images to creating content that excels across **all major aesthetic dimensions simultaneously**! 🎨✨🧠

### 🌟 The Multi-Head Advantage:
Rather than hoping one metric captures "good aesthetics," you explicitly optimize for:
- **Composition mastery** (Spatial)
- **Cultural richness** (Iconographic)  
- **Artistic excellence** (Style)
- **Technical perfection** (Fidelity)
- **Material realism** (Material)

The result: Images that satisfy expert aesthetic criteria across **every dimension that matters**! 🚀
