# Makefile for SDXL DoRA Trainer - RLHF Pipeline
# Usage: make <target>

.PHONY: help install generate-images-dual generate-images-single train-reward train-reward-multimodal train-dspo annotate clean

# Default variables
PROMPTS_FILE := data/prompts.csv
OUTPUT_DIR := data/images
DORA_WEIGHTS := outputs/dora_weights
MODEL := stabilityai/stable-diffusion-xl-base-1.0
SEEDS := 0 1
WIDTH := 1024
HEIGHT := 1024
STEPS := 50
FUSION_METHOD := concat

help:
	@echo "SDXL DoRA RLHF Training Pipeline"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@echo "  generate-images-dual   - Generate images using 2 GPUs (fastest)"
	@echo "  generate-images-single - Generate images using 1 GPU"
	@echo "  train-reward          - Train image-only reward model on ratings"
	@echo "  train-reward-multimodal - Train multimodal (image+prompt) reward model"
	@echo "  train-dspo            - Train DSPO model using reward"
	@echo "  annotate              - Create annotation pairs for experts"
	@echo "  install               - Install dependencies"
	@echo "  clean                 - Clean generated files"
	@echo "  explain-workflow      - Explain the complete RLHF workflow"
	@echo "  explain-multimodal    - Explain multimodal training benefits"
	@echo ""
	@echo "Configuration (override with VAR=value):"
	@echo "  PROMPTS_FILE = $(PROMPTS_FILE)"
	@echo "  OUTPUT_DIR   = $(OUTPUT_DIR)"
	@echo "  DORA_WEIGHTS = $(DORA_WEIGHTS)"
	@echo "  MODEL        = $(MODEL)"
	@echo "  SEEDS        = $(SEEDS)"
	@echo "  WIDTH        = $(WIDTH)"
	@echo "  HEIGHT       = $(HEIGHT)"
	@echo "  STEPS        = $(STEPS)"
	@echo "  FUSION_METHOD = $(FUSION_METHOD) (concat, add, cross_attention)"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

generate-images-dual:
	@echo "🚀 Generating images using dual GPU setup..."
	@echo "Using prompts: $(PROMPTS_FILE)"
	@echo "Output directory: $(OUTPUT_DIR)"
	python scripts/generate_dual_gpu.py \
		--prompts $(PROMPTS_FILE) \
		--output $(OUTPUT_DIR) \
		--model $(MODEL) \
		--dora-weights $(DORA_WEIGHTS) \
		--width $(WIDTH) \
		--height $(HEIGHT) \
		--steps $(STEPS) \
		--seeds $(SEEDS)
	@echo "✅ Dual GPU image generation complete!"

generate-images-single:
	@echo "🖼️ Generating images using single GPU..."
	python scripts/generate_images.py \
		--prompts $(PROMPTS_FILE) \
		--output $(OUTPUT_DIR) \
		--model $(MODEL) \
		--device cuda \
		--dora-weights $(DORA_WEIGHTS) \
		--width $(WIDTH) \
		--height $(HEIGHT) \
		--steps $(STEPS) \
		--seeds $(SEEDS)
	@echo "✅ Single GPU image generation complete!"

annotate:
	@echo "📝 Setting up multi-head rating annotation (5 aspects)..."
	@echo "Creating template for expert ratings across specialized aspects:"
	@echo "  - Spatial: Composition, layout, rule of thirds, balance"
	@echo "  - Iconographic: Symbols, cultural elements, narrative content"
	@echo "  - Style: Artistic technique, color harmony, aesthetic movement"
	@echo "  - Fidelity: Technical quality, sharpness, noise, artifacts"
	@echo "  - Material: Texture realism, surface properties, lighting"
	python scripts/build_pairs.py \
		--images-csv data/generated.csv \
		--output data/annotations/multihead_ratings.csv \
		--format multihead
	@echo "✅ Multi-head annotation template created!"
	@echo "📋 Next: Have experts rate each image on all 5 aspects (0-10 scale)"
	@echo "� Format: prompt_id,image_path,spatial_rating,icono_rating,style_rating,fidelity_rating,material_rating"

train-reward:
	@echo "🏆 Training image-only multi-head reward model (5 specialized heads)..."
	@echo "Using multi-aspect ratings from: data/annotations/multihead_ratings.csv"
	python scripts/train_reward.py \
		--mode multihead \
		--ratings data/annotations/multihead_ratings.csv \
		--output outputs/multihead_reward_model \
		--epochs 15 \
		--batch-size 4 \
		--learning-rate 1e-4 \
		--min-rating-diff 0.5
	@echo "✅ Image-only multi-head reward model training complete!"
	@echo "📊 Check individual head performance in outputs/multihead_reward_model/metrics.json"

train-reward-multimodal:
	@echo "🏆🔤 Training multimodal (image+prompt) reward model..."
	@echo "Enhanced with prompt-image alignment assessment!"
	@echo "Using ratings: data/annotations/multihead_ratings.csv"
	@echo "Using prompts: $(PROMPTS_FILE)"
	@echo "Fusion method: $(FUSION_METHOD)"
	python scripts/train_multimodal_reward.py \
		--mode multihead \
		--ratings data/annotations/multihead_ratings.csv \
		--prompts $(PROMPTS_FILE) \
		--output outputs/multimodal_reward_model_$(FUSION_METHOD) \
		--epochs 15 \
		--batch-size 4 \
		--learning-rate 1e-4 \
		--min-rating-diff 0.5 \
		--fusion-method $(FUSION_METHOD) \
		--mixed-precision bf16
	@echo "✅ Multimodal reward model training complete!"
	@echo "🎯 Model now assesses prompt-image alignment + visual quality!"
	@echo "📊 Check head performance and fusion effectiveness in outputs/"

# Quick multimodal training with different fusion methods
train-multimodal-concat:
	$(MAKE) train-reward-multimodal FUSION_METHOD=concat

train-multimodal-attention:
	$(MAKE) train-reward-multimodal FUSION_METHOD=cross_attention

train-multimodal-add:
	$(MAKE) train-reward-multimodal FUSION_METHOD=add

train-dspo:
	@echo "🎯 Training DSPO with multi-head reward feedback..."
	@echo "Using 5-aspect reward model for fine-grained optimization..."
	python scripts/train_dspo.py \
		--base-model $(MODEL) \
		--reward-model outputs/multihead_reward_model \
		--dataset data/annotations/multihead_ratings.csv \
		--output outputs/dspo_model \
		--epochs 5 \
		--batch-size 4 \
		--use-multihead-rewards
	@echo "✅ DSPO training with multi-head feedback complete!"
	@echo "🎨 Model now optimized across all 5 aesthetic aspects!"
	@echo "✅ DSPO training with multi-rating feedback complete!"

# Convenience targets for the full pipeline
pipeline-step1: generate-images-dual annotate
	@echo "🎉 Step 1 complete! Images generated and pairs created."
	@echo "📋 Next: Have experts rate pairs, then run 'make pipeline-step2' or 'make pipeline-step2-multimodal'"

pipeline-step2: train-reward train-dspo
	@echo "🎉 Image-only RLHF pipeline complete!"
	@echo "🚀 Your improved model is in outputs/dspo_model"

pipeline-step2-multimodal: train-reward-multimodal train-dspo
	@echo "🎉 Multimodal RLHF pipeline complete!"
	@echo "🚀 Your prompt-aligned model is in outputs/dspo_model"
	@echo "🎯 Model optimized for prompt-image alignment + visual quality!"

# Compare training approaches
compare-models: train-reward train-reward-multimodal
	@echo "🔬 Training comparison complete!"
	@echo "📊 Compare image-only vs multimodal models:"
	@echo "   Image-only: outputs/multihead_reward_model/"
	@echo "   Multimodal: outputs/multimodal_reward_model_$(FUSION_METHOD)/"

# Quick test with small dataset
test-pipeline:
	@echo "🧪 Running test pipeline with small dataset..."
	$(MAKE) generate-images-dual PROMPTS_FILE=data/prompts.csv OUTPUT_DIR=test_output STEPS=20
	$(MAKE) annotate OUTPUT_DIR=test_output
	@echo "✅ Test pipeline complete! Check test_output/"

test-multimodal:
	@echo "🧪 Testing multimodal training with small dataset..."
	$(MAKE) train-reward-multimodal PROMPTS_FILE=data/prompts.csv FUSION_METHOD=concat
	@echo "✅ Multimodal test complete!"

validate-multimodal:
	@echo "🔍 Validating multimodal components..."
	@python -c "import sys; sys.path.append('.'); from dspo.multimodal_reward import MultimodalMultiHeadReward; import torch; model = MultimodalMultiHeadReward(fusion_method='concat'); dummy_images = torch.randn(2, 3, 224, 224); dummy_prompts = ['A sunset', 'A cat']; outputs = model(dummy_images, dummy_prompts); print('✅ Multimodal model validation successful!')"
	@echo "✅ Multimodal validation complete!"

clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf data/images/*
	rm -rf outputs/*
	rm -rf test_output/
	rm -f data/annotations/pairs.jsonl
	rm -f data/generated*.csv
	@echo "✅ Cleanup complete!"

# Advanced targets
benchmark-gpu:
	@echo "📊 Benchmarking GPU performance..."
	@echo "Attempting dual GPU generation..."
	-python scripts/generate_dual_gpu.py \
		--prompts data/prompts.csv \
		--output benchmark_output \
		--steps 20 \
		--seeds 0 || \
	(echo "⚠️ Dual GPU failed, falling back to single GPU..." && \
	 python scripts/generate_images.py \
		--prompts data/prompts.csv \
		--output benchmark_output \
		--model $(MODEL) \
		--device cuda \
		--width $(WIDTH) \
		--height $(HEIGHT) \
		--steps 20 \
		--seeds 0)
	@echo "Check GPU utilization with: nvidia-smi -l 1"

monitor-training:
	@echo "📈 Monitoring training progress..."
	watch -n 5 'tail -20 outputs/*/training.log'

# Help for workflow understanding
explain-workflow:
	@echo "🔄 RLHF (Reinforcement Learning from Human Feedback) Workflow:"
	@echo ""
	@echo "1. IMAGE GENERATION (generate-images-dual):"
	@echo "   - Generate 2 images per prompt with different seeds"
	@echo "   - Uses SDXL base model + optional DoRA weights"
	@echo "   - Output: data/images/ with metadata in generated_dual_gpu.csv"
	@echo ""
	@echo "2. ANNOTATION SETUP (annotate):"
	@echo "   - Create rating templates for expert annotation"
	@echo "   - Each image gets individual ratings (not pairwise comparison)"
	@echo "   - Output: data/annotations/multihead_ratings.csv template"
	@echo ""
	@echo "3. HUMAN ANNOTATION (manual step):"
	@echo "   - Experts rate each image individually across 5 specialized aspects:"
	@echo "     * Spatial (0-10): Composition, layout, rule of thirds, balance"
	@echo "     * Iconographic (0-10): Symbols, cultural elements, narrative"
	@echo "     * Style (0-10): Artistic technique, color harmony, aesthetic"
	@echo "     * Fidelity (0-10): Technical quality, sharpness, artifacts"
	@echo "     * Material (0-10): Texture realism, surface properties"
	@echo "   - Multiple experts rate same images for reliability"
	@echo "   - Format: prompt_id,image_path,spatial_rating,icono_rating,style_rating,fidelity_rating,material_rating"
	@echo "   - Output: data/annotations/multihead_ratings.csv"
	@echo ""
	@echo "4. REWARD MODEL TRAINING (train-reward):"
	@echo "   - Train 5 specialized heads + combination weights"
	@echo "   - Each head learns one aesthetic aspect"
	@echo "   - Uses frozen OpenCLIP ViT-L/14 backbone"
	@echo "   - Input: multihead_ratings.csv with numerical scores"
	@echo "   - Output: outputs/multihead_reward_model/"
	@echo ""
	@echo "5. DSPO TRAINING (train-dspo):"
	@echo "   - Fine-tune SDXL using multi-head reward model feedback"
	@echo "   - Each head provides specialized optimization signals"
	@echo "   - Uses Direct Statistical Preference Optimization"
	@echo "   - Learns to optimize all 5 aesthetic aspects simultaneously"
	@echo "   - Output: outputs/dspo_model/ (improved SDXL)"
	@echo ""
	@echo "💡 The result is an SDXL model optimized across 5 specialized aesthetic dimensions!"

explain-multimodal:
	@echo "🔤🖼️ MULTIMODAL REWARD MODEL TRAINING:"
	@echo ""
	@echo "🎯 ENHANCED APPROACH - Image + Prompt Processing:"
	@echo "   Instead of just analyzing images, the multimodal model processes:"
	@echo "   • Visual content (images) - same as before"
	@echo "   • Text prompts - understanding what the image should show"
	@echo "   • Combined analysis - how well images match their prompts"
	@echo ""
	@echo "🧠 FUSION STRATEGIES:"
	@echo "   concat (default):"
	@echo "     - Simple concatenation of image and text features"
	@echo "     - Reliable and fast, good starting point"
	@echo "     - Usage: make train-multimodal-concat"
	@echo ""
	@echo "   cross_attention (advanced):"
	@echo "     - Attention-based fusion for sophisticated interaction"
	@echo "     - Better prompt-image alignment understanding"
	@echo "     - Usage: make train-multimodal-attention"
	@echo ""
	@echo "   add (simple):"
	@echo "     - Element-wise addition of features"
	@echo "     - Lightweight but requires same dimensions"
	@echo "     - Usage: make train-multimodal-add"
	@echo ""
	@echo "🎯 SPECIALIZED HEADS WITH PROMPT CONTEXT:"
	@echo "   Spatial: Composition matching prompt descriptions"
	@echo "   Iconographic: Elements aligning with prompt content"
	@echo "   Style: Artistic style consistency with prompt cues"
	@echo "   Fidelity: Image quality relative to prompt complexity"
	@echo "   Material: Properties as described in prompts"
	@echo ""
	@echo "💡 BENEFITS FOR SDXL TRAINING:"
	@echo "   ✅ Direct prompt-image alignment optimization"
	@echo "   ✅ Context-aware quality assessment"
	@echo "   ✅ Better generalization across prompt types"
	@echo "   ✅ Enhanced understanding of text-visual relationships"
	@echo "   ✅ Critical for prompt-conditioned generation"
	@echo ""
	@echo "🚀 USAGE EXAMPLES:"
	@echo "   Basic multimodal:     make train-reward-multimodal"
	@echo "   With attention:       make train-reward-multimodal FUSION_METHOD=cross_attention"
	@echo "   Compare approaches:   make compare-models"
	@echo "   Full multimodal flow: make pipeline-step2-multimodal"
	@echo ""
	@echo "📊 MONITORING:"
	@echo "   • Per-head F1 scores show individual aspect performance"
	@echo "   • Head importance weights reveal learned priorities"
	@echo "   • Fusion effectiveness visible in combined metrics"
