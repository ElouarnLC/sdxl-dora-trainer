# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Debug tools for analyzing DoRA weights
- Fix tool for corrupted DoRA weights
- Comprehensive documentation structure
- Installation guide with multiple methods
- API reference documentation
- Contributing guidelines

### Changed
- Default mixed precision changed from fp16 to "no" for stability
- Reorganized repository structure with proper directories
- Updated README with black image fix prominence
- Improved error messages and warnings

### Fixed
- **CRITICAL**: Black image generation caused by fp16 mixed precision
- NaN/Inf value handling in DoRA weights
- Better parameter validation during training setup
- More stable training defaults

## [1.0.0] - 2025-01-XX

### Added
- Initial release of SDXL DoRA Trainer
- DoRA (Weight-Decomposed Low-Rank Adaptation) support
- Production-ready training pipeline
- Comprehensive configuration system
- Multi-GPU training support via Accelerate
- TensorBoard and Weights & Biases integration
- Rich terminal interface with progress tracking
- Automated environment setup script
- Dataset validation and analysis tools
- Inference pipeline for trained models
- Memory optimization features
- Gradient checkpointing support
- 8-bit Adam optimizer support
- Cross-platform compatibility (Windows, Linux, macOS)

### Features
- **Training**
  - DoRA fine-tuning for SDXL models
  - Configurable rank and alpha parameters
  - Custom target modules selection
  - Learning rate scheduling
  - Gradient accumulation
  - Mixed precision training
  - Validation image generation
  - Checkpoint saving and resuming

- **Inference**
  - Easy model loading with DoRA weights
  - Batch image generation
  - Interactive generation mode
  - Configurable generation parameters
  - Memory-efficient processing

- **Utilities**
  - Environment validation
  - GPU memory checking
  - Dataset analysis and statistics
  - Parameter suggestion system
  - Configuration management
  - Error diagnostics

- **Quality of Life**
  - Beautiful terminal interface
  - Comprehensive logging
  - Progress tracking
  - Error handling and recovery
  - Detailed documentation
  - Example configurations

### Technical Details
- **Supported Models**: Stable Diffusion XL base models
- **Memory Requirements**: 12GB+ VRAM recommended
- **Python Versions**: 3.8, 3.9, 3.10, 3.11
- **CUDA Support**: 11.8 and 12.1
- **Precision Modes**: fp32, fp16, bf16 (with stability warnings)

### Documentation
- Complete installation guide
- Quick start tutorial
- Configuration reference
- API documentation
- Troubleshooting guide
- Contributing guidelines
- Example workflows

---

## Version History Notes

### Naming Convention
- Major versions (x.0.0): Significant feature additions or breaking changes
- Minor versions (x.y.0): New features, improvements, backward compatible
- Patch versions (x.y.z): Bug fixes, documentation updates

### Support Policy
- **Current Version**: Full support with updates and bug fixes
- **Previous Major**: Security fixes and critical bug fixes only
- **Older Versions**: Community support only

### Migration Guides

#### From LoRA to DoRA
If migrating from traditional LoRA fine-tuning:
1. Update your configuration to use DoRA-specific parameters
2. Adjust rank and alpha values (DoRA typically uses lower values)
3. Be aware that DoRA weights are not compatible with LoRA weights
4. Retrain your models using the DoRA trainer

#### Mixed Precision Migration
If you were using fp16 and experiencing black images:
1. Change `mixed_precision` from "fp16" to "no" in your config
2. Expect slightly slower training but much more stable results
3. Consider "bf16" if you have an Ampere+ GPU for better speed/stability balance
4. Existing fp16-trained weights may need to be retrained for best quality

### Breaking Changes

#### v1.0.0
- Initial release, no breaking changes from previous versions

### Deprecation Notices

Currently no deprecated features.

### Known Issues

- fp16 mixed precision can cause numerical instability with DoRA (documented fix available)
- Memory usage may be higher than expected on some systems (optimization ongoing)
- Some older GPUs may require specific CUDA versions for optimal performance

### Future Roadmap

#### Planned Features
- Support for SDXL Turbo and Lightning models
- Advanced regularization techniques
- Custom dataset augmentation options
- Model merging utilities
- Quantization support for inference
- ControlNet DoRA fine-tuning
- Improved memory optimization
- Web interface for easier use

#### Under Consideration
- Support for other diffusion models (SD 1.5, SD 2.x)
- Automatic hyperparameter tuning
- Multi-node training support
- Integration with popular ML platforms
- Custom scheduler implementations
- Advanced monitoring and profiling tools

---

For detailed information about any release, see the corresponding GitHub release notes and documentation updates.
