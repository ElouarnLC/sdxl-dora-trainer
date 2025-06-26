# 🚨 Important: Black Image Issue with DoRA Training

## Problem
When training SDXL models with DoRA, you may encounter **black/empty images** during generation. This is a known issue with a simple fix.

## Root Cause
The issue is caused by **mixed precision training (fp16)** creating NaN (Not a Number) or infinite values in the DoRA weights. These invalid values propagate through the diffusion process and cause the image generation to fail, resulting in black images.

### Error Messages You Might See:
```
RuntimeWarning: invalid value encountered in cast
  images = (images * 255).round().astype("uint8")
WARNING  Image 0 appears to be mostly black (mean: 0.00)
```

## ✅ Solution

**Use `--mixed_precision no` instead of `fp16`:**

```bash
# ❌ This often causes black images:
python sdxl_dora_trainer.py --mixed_precision fp16 --dataset_path ./data

# ✅ This works reliably:
python sdxl_dora_trainer.py --mixed_precision no --dataset_path ./data

# ✅ Alternative if you have a modern GPU (Ampere+):
python sdxl_dora_trainer.py --mixed_precision bf16 --dataset_path ./data
```

## Why This Happens

1. **DoRA (Weight-Decomposed LoRA)** involves complex weight decomposition and magnitude scaling
2. **fp16 has limited precision** (~3-4 significant digits) compared to fp32 (~7 significant digits)
3. **Small numerical errors accumulate** during training and cause weight values to become NaN/Inf
4. **NaN values propagate** through the entire generation process
5. **Image decoding fails** when trying to convert NaN latents to pixels
6. **Result:** Black images (all zeros)

## Performance Impact

| Mixed Precision | Speed | Memory | Stability | Recommendation |
|----------------|-------|---------|-----------|----------------|
| `fp16` | Fastest | Lowest | ⚠️ Unstable with DoRA | ❌ Not recommended |
| `bf16` | Fast | Low | ✅ Good | ✅ Use if you have Ampere+ GPU |
| `no` | Slower | Higher | ✅ Very stable | ✅ Most reliable choice |

## Additional Tips

1. **Check your GPU**: Modern GPUs (RTX 30xx, 40xx, A100, etc.) support `bf16` which is more stable than `fp16`
2. **Monitor training**: If you see loss becoming NaN, switch to `no` immediately
3. **Validate weights**: Use the debug tools to check for NaN/Inf values in saved checkpoints
4. **Conservative settings**: Lower learning rates (5e-5) help prevent numerical instability

## For Existing Corrupted Weights

If you already have DoRA weights that produce black images:

```bash
# Analyze the weights
python debug_dora_weights.py --weights_path ./output/checkpoints/checkpoint-250

# Fix corrupted weights
python fix_dora_weights.py --weights_path ./output/checkpoints/checkpoint-250
```

## Default Configuration

This trainer now defaults to `mixed_precision = "no"` to prevent this issue. The slight performance cost is worth the stability for DoRA training.
