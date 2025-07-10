# Makefile for Diffusion RLHF Pipeline

.PHONY: all generate annotate build-pairs train-reward train-dspo sweep clean test

# Default target
all: generate build-pairs train-reward train-dspo

# Generate images from prompts
generate:
	python scripts/generate_images.py --prompts data/prompts.csv --output data/images/

# Placeholder for manual annotation step
annotate:
	@echo "Manual annotation required: Create data/annotations/ratings.csv"
	@echo "Format: prompt_id,seed,rating"
	@echo "Example entries:"
	@echo "0,0,4.5"
	@echo "0,1,3.2"

# Build preference pairs from ratings
build-pairs:
	python scripts/build_pairs.py --ratings data/annotations/ratings.csv --output data/annotations/pairs.jsonl

# Train reward model
train-reward:
	python scripts/train_reward.py --pairs data/annotations/pairs.jsonl --output outputs/reward_model/

# Train DSPO model
train-dspo:
	python scripts/train_dspo.py --reward outputs/reward_model/ --pairs data/annotations/pairs.jsonl --output outputs/dspo_model/

# Run hyperparameter sweep
sweep:
	python scripts/sweep_dspo_optuna.py --reward outputs/reward_model/ --pairs data/annotations/pairs.jsonl --output outputs/sweep/

# Run tests
test:
	pytest tests/ -v

# Clean outputs
clean:
	rm -rf outputs/
	rm -rf data/images/*
	rm -f data/annotations/pairs.jsonl

# Setup environment
setup:
	pip install -r requirements.txt
	mkdir -p data/images data/annotations outputs/samples

# Development setup
dev-setup: setup
	pip install -e .
	pre-commit install
