---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: 'bug'
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Run command `...`
2. Set parameter `...`
3. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Screenshots/Logs
If applicable, add screenshots or paste error logs to help explain your problem.

```
Paste error logs here
```

## Environment
Please complete the following information:
- **OS**: [e.g., Windows 10, Ubuntu 20.04, macOS 12]
- **Python version**: [e.g., 3.10.8]
- **GPU**: [e.g., RTX 4090 24GB]
- **CUDA version**: [e.g., 12.1]
- **Package versions**:
  ```bash
  # Run this command and paste the output:
  python -c "import torch, diffusers, peft, transformers; print(f'torch: {torch.__version__}, diffusers: {diffusers.__version__}, peft: {peft.__version__}, transformers: {transformers.__version__}')"
  ```

## Configuration
If using a config file, please paste the relevant parts:
```yaml
# Paste your configuration here
```

## Additional Context
Add any other context about the problem here.

## Checklist
- [ ] I have searched existing issues
- [ ] I have run `python utils.py check-env` to verify my environment
- [ ] I can reproduce this issue consistently
- [ ] I have checked the [BLACK_IMAGE_FIX.md](../docs/BLACK_IMAGE_FIX.md) if experiencing black images
