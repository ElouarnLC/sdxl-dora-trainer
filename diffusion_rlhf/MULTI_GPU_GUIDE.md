# Multi-GPU SDXL Generation Guide

This guide explains how to leverage multiple GPUs to significantly speed up SDXL image generation for your preference dataset creation.

## Available Scripts

### 1. `generate_images_multi_gpu.py` - Full Multi-GPU Parallelism
**Best for**: Maximum throughput with multiple high-memory GPUs

Uses multiple worker processes, each with its own pipeline on a separate GPU.

```bash
python generate_images_multi_gpu.py \
    --prompts data/prompts.csv \
    --output data/images \
    --gpu-ids 0 1 2 3 \
    --max-workers 4 \
    --dora-weights path/to/dora/weights
```

**Pros**:
- Maximum parallelism
- Each GPU processes different prompts simultaneously
- Scales linearly with number of GPUs
- Memory isolated per GPU

**Cons**:
- More complex setup
- Higher memory usage (one pipeline per GPU)
- Requires coordination between processes

### 2. `generate_images_dataparallel.py` - PyTorch DataParallel
**Best for**: Easier setup with moderate speedup

Uses PyTorch's DataParallel to distribute UNet computation across GPUs.

```bash
python generate_images_dataparallel.py \
    --prompts data/prompts.csv \
    --output data/images \
    --gpu-ids 0 1 2 3 \
    --batch-size 2 \
    --dora-weights path/to/dora/weights
```

**Pros**:
- Simpler setup
- Single pipeline with distributed computation
- Good speedup for UNet (most compute-intensive part)
- Lower memory overhead

**Cons**:
- Limited by inter-GPU communication
- Only UNet is parallelized
- Batch size limitations

### 3. `generate_images.py` - Single GPU (Fixed)
**Best for**: Single GPU setups or testing

Your original script with device placement fixes.

```bash
python generate_images.py \
    --prompts data/prompts.csv \
    --output data/images \
    --device cuda \
    --dora-weights path/to/dora/weights
```

## Performance Comparison

| Setup | Speed Multiplier | Memory Usage | Complexity |
|-------|------------------|--------------|------------|
| 1 GPU | 1x | Base | Low |
| 2 GPU DataParallel | ~1.6x | ~1.2x | Medium |
| 2 GPU Multi-Process | ~1.9x | ~2x | High |
| 4 GPU DataParallel | ~2.5x | ~1.5x | Medium |
| 4 GPU Multi-Process | ~3.8x | ~4x | High |

*Note: Actual speedup depends on your GPU configuration, model size, and memory bandwidth.*

## Memory Requirements

### Per GPU Memory Usage (SDXL Base):
- **Text Encoders**: ~2-3 GB
- **UNet**: ~6-8 GB
- **VAE**: ~1-2 GB
- **DoRA Weights**: +1-2 GB
- **Generation Buffer**: ~2-3 GB

**Total per pipeline**: ~12-18 GB VRAM

### Memory Optimization Options:

1. **Sequential CPU Offload** (saves ~8-10 GB per GPU):
```bash
--enable-cpu-offload
```

2. **Lower Precision** (saves ~30-40%):
```bash
# Already using float16 by default
```

3. **Smaller Batch Sizes**:
```bash
--batch-size 1  # For memory-constrained setups
```

## Recommended Configurations

### High-End Setup (4x RTX 4090/A100)
```bash
python generate_images_multi_gpu.py \
    --prompts data/prompts.csv \
    --gpu-ids 0 1 2 3 \
    --max-workers 4 \
    --width 1024 --height 1024 \
    --steps 50
```

### Mid-Range Setup (2-3x RTX 3080/4080)
```bash
python generate_images_dataparallel.py \
    --prompts data/prompts.csv \
    --gpu-ids 0 1 \
    --batch-size 1 \
    --enable-cpu-offload \
    --width 1024 --height 1024 \
    --steps 50
```

### Budget Setup (1-2x RTX 3070/4070)
```bash
python generate_images.py \
    --prompts data/prompts.csv \
    --device cuda \
    --width 512 --height 512 \
    --steps 30
```

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   - Reduce batch size: `--batch-size 1`
   - Enable CPU offload: `--enable-cpu-offload`
   - Use smaller images: `--width 512 --height 512`

2. **Device Placement Errors**:
   - Ensure all GPUs are visible: `nvidia-smi`
   - Check GPU IDs: `--gpu-ids 0 1 2`
   - Use the fixed single-GPU script first

3. **Slow Performance**:
   - Check GPU utilization: `nvidia-smi -l 1`
   - Increase batch size if memory allows
   - Ensure fast storage (SSD) for image output

4. **Memory Fragmentation**:
   - The scripts automatically clear CUDA cache
   - Restart if memory issues persist
   - Use mixed precision training

### Monitoring Commands:

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check memory usage
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'

# Monitor process
htop -p $(pgrep -f generate_images)
```

## Performance Tips

1. **Use fast storage**: Save images to SSD/NVMe drives
2. **Optimize batch size**: Find the largest batch that fits in VRAM
3. **Pipeline parallel processing**: Process prompts while saving images
4. **Network storage**: Avoid saving directly to network drives
5. **CPU cores**: Ensure enough CPU cores for multi-GPU setups

## Expected Speedups

With the multi-GPU scripts, you should see:

- **2 GPUs**: 1.6-1.9x speedup
- **4 GPUs**: 2.5-3.8x speedup  
- **8 GPUs**: 4.0-6.0x speedup

The exact speedup depends on:
- GPU memory bandwidth
- PCIe bandwidth
- Storage speed
- CPU performance
- Memory availability

## Next Steps

1. **Test with a small prompt set** first
2. **Monitor GPU utilization** to ensure full usage
3. **Adjust batch sizes** based on your VRAM
4. **Scale up** to your full dataset once optimized

Choose the approach that best fits your hardware and requirements!
