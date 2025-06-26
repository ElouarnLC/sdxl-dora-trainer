# Contributing to SDXL DoRA Trainer

Thank you for your interest in contributing to SDXL DoRA Trainer! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting Changes](#submitting-changes)
- [Style Guide](#style-guide)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project follows a Code of Conduct to ensure a welcoming environment for all contributors. Please be respectful, inclusive, and professional in all interactions.

## Getting Started

### Types of Contributions

We welcome various types of contributions:

- **Bug Reports**: Report issues or unexpected behavior
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit bug fixes or new features
- **Documentation**: Improve or expand documentation
- **Testing**: Add or improve test coverage
- **Examples**: Provide usage examples or tutorials

### Before You Start

1. Check existing [issues](https://github.com/ElouarnLC/sdxl-dora-trainer/issues) and [pull requests](https://github.com/ElouarnLC/sdxl-dora-trainer/pulls)
2. For major changes, open an issue first to discuss the proposed changes
3. Make sure you have the necessary development environment set up

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- NVIDIA GPU (recommended for testing)

### Setup Steps

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/ElouarnLC/sdxl-dora-trainer.git
   cd sdxl-dora-trainer
   ```

2. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/ElouarnLC/sdxl-dora-trainer.git
   ```

3. **Create development environment**
   ```bash
   # Using conda (recommended)
   conda create -n sdxl-dora-dev python=3.10
   conda activate sdxl-dora-dev
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate  # Windows
   ```

4. **Install dependencies**
   ```bash
   # Install development dependencies
   pip install -r requirements-dev.txt
   
   # Install in editable mode
   pip install -e .
   ```

5. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Making Changes

### Workflow

1. **Update your fork**
   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number
   ```

3. **Make your changes**
   - Write clear, concise code
   - Follow the style guide
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   # Run tests
   python -m pytest tests/
   
   # Run specific test
   python -m pytest tests/test_specific.py
   
   # Run with coverage
   python -m pytest --cov=sdxl_dora_trainer tests/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

### Commit Message Guidelines

Use conventional commit format:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring
- `style:` for formatting changes
- `chore:` for maintenance tasks

Examples:
```
feat: add support for SDXL-Turbo models
fix: resolve black image issue with fp16 precision
docs: update installation instructions
test: add unit tests for DoRA weight validation
```

## Submitting Changes

### Pull Request Process

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**
   - Go to GitHub and create a PR from your branch
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Reference any related issues

3. **PR Template**
   ```markdown
   ## Description
   Brief description of changes made.
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   - [ ] Other (specify)
   
   ## Testing
   - [ ] Tests pass locally
   - [ ] Added new tests
   - [ ] Manual testing completed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No breaking changes (or documented)
   ```

### Review Process

1. **Automated Checks**
   - Code formatting (black, isort)
   - Linting (flake8, mypy)
   - Tests (pytest)
   - Security checks

2. **Human Review**
   - Code quality and clarity
   - Test coverage
   - Documentation completeness
   - Performance considerations

3. **Feedback Incorporation**
   - Address reviewer comments
   - Make requested changes
   - Update tests if needed

## Style Guide

### Python Code Style

We use the following tools for code formatting:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

### Configuration

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
```

### Code Standards

1. **Formatting**
   ```bash
   # Format code
   black .
   isort .
   ```

2. **Type Hints**
   ```python
   def train_model(config: TrainingConfig) -> None:
       """Train the model with given configuration."""
       pass
   ```

3. **Docstrings**
   ```python
   def generate_images(
       prompts: List[str], 
       steps: int = 50
   ) -> List[Image.Image]:
       """Generate images from text prompts.
       
       Args:
           prompts: List of text prompts
           steps: Number of inference steps
           
       Returns:
           List of generated PIL Images
           
       Raises:
           ValueError: If prompts list is empty
       """
       pass
   ```

4. **Error Handling**
   ```python
   try:
       result = risky_operation()
   except SpecificError as e:
       logger.error(f"Operation failed: {e}")
       raise
   ```

## Testing

### Test Structure

```
tests/
├── unit/
│   ├── test_config.py
│   ├── test_trainer.py
│   └── test_utils.py
├── integration/
│   ├── test_training_pipeline.py
│   └── test_inference_pipeline.py
└── fixtures/
    ├── sample_dataset/
    └── test_config.yaml
```

### Writing Tests

1. **Unit Tests**
   ```python
   import pytest
   from sdxl_dora_trainer import TrainingConfig
   
   def test_config_validation():
       """Test configuration validation."""
       config = TrainingConfig(dataset_path="./test")
       assert config.rank > 0
       assert config.alpha > 0
   ```

2. **Integration Tests**
   ```python
   def test_training_pipeline(tmp_path):
       """Test complete training pipeline."""
       config = TrainingConfig(
           dataset_path=str(tmp_path / "dataset"),
           output_dir=str(tmp_path / "output"),
           max_train_steps=5
       )
       trainer = DoRATrainer(config)
       trainer.train()
       assert (tmp_path / "output" / "checkpoints").exists()
   ```

3. **Fixtures**
   ```python
   @pytest.fixture
   def sample_config():
       """Provide sample configuration for testing."""
       return TrainingConfig(
           dataset_path="./test_dataset",
           rank=32,
           alpha=16
       )
   ```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=sdxl_dora_trainer

# Run specific test file
python -m pytest tests/test_trainer.py

# Run with verbose output
python -m pytest -v

# Run only fast tests
python -m pytest -m "not slow"
```

## Documentation

### Types of Documentation

1. **Code Documentation**
   - Docstrings for functions and classes
   - Inline comments for complex logic
   - Type hints

2. **User Documentation**
   - README.md updates
   - Tutorial additions
   - API reference updates

3. **Developer Documentation**
   - Architecture decisions
   - Contributing guidelines
   - Setup instructions

### Documentation Guidelines

1. **Write for your audience**
   - User docs: focus on how to use features
   - Developer docs: focus on how things work

2. **Keep it current**
   - Update docs when code changes
   - Remove outdated information

3. **Use examples**
   - Provide code examples
   - Include expected outputs
   - Show common use cases

### Building Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation
cd docs
make html

# Serve locally
python -m http.server 8000
```

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
Clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. Set parameter '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Windows 10]
- Python version: [e.g., 3.10]
- GPU: [e.g., RTX 4090]
- CUDA version: [e.g., 12.1]

**Additional Context**
Any other context or screenshots.
```

### Feature Requests

Use the feature request template:

```markdown
**Feature Description**
Clear description of the feature.

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives**
Other solutions you've considered.

**Additional Context**
Any other context or examples.
```

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):
- MAJOR.MINOR.PATCH
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Checklist

1. Update version number
2. Update CHANGELOG.md
3. Create release notes
4. Tag release
5. Build and test
6. Deploy to PyPI (if applicable)

## Getting Help

- **Documentation**: Check existing docs first
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions for questions
- **Community**: Join our community channels

Thank you for contributing to SDXL DoRA Trainer! 🎉
