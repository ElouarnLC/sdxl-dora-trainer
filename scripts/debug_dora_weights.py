#!/usr/bin/env python3
"""
Debug script to analyze DoRA weights and identify issues causing black images.
"""

import torch
import numpy as np
from pathlib import Path
from peft import PeftModel
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def analyze_dora_weights(weights_path: str):
    """Analyze DoRA weights for issues that could cause black images."""
    console.print(Panel("[bold blue]Analyzing DoRA Weights[/bold blue]"))
    
    weights_path = Path(weights_path)
    if not weights_path.exists():
        console.print(f"[red]Error: Weights path does not exist: {weights_path}[/red]")
        return False
    
    try:
        # Load base UNet
        console.print("Loading base UNet model...")
        base_unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="unet",
            torch_dtype=torch.float16
        )
        
        # Load DoRA weights
        console.print(f"Loading DoRA weights from {weights_path}...")
        dora_unet = PeftModel.from_pretrained(base_unet, weights_path)
        
        # Analyze weights
        console.print("\nAnalyzing weight statistics...")
        
        # Create analysis table
        table = Table(title="DoRA Weight Analysis")
        table.add_column("Parameter", style="cyan")
        table.add_column("Shape", style="magenta")
        table.add_column("Min", style="green")
        table.add_column("Max", style="green")
        table.add_column("Mean", style="yellow")
        table.add_column("Std", style="yellow")
        table.add_column("NaN Count", style="red")
        table.add_column("Inf Count", style="red")
        table.add_column("Zero %", style="blue")
        
        total_params = 0
        total_nan = 0
        total_inf = 0
        problematic_params = []
        
        for name, param in dora_unet.named_parameters():
            if 'lora_' in name or 'dora_' in name:  # Only analyze DoRA/LoRA parameters
                param_data = param.detach().cpu().numpy()
                total_params += param.numel()
                
                # Count NaN and Inf
                nan_count = np.isnan(param_data).sum()
                inf_count = np.isinf(param_data).sum()
                zero_percent = (param_data == 0).sum() / param_data.size * 100
                
                total_nan += nan_count
                total_inf += inf_count
                
                # Check for problematic parameters
                if nan_count > 0 or inf_count > 0:
                    problematic_params.append(name)
                
                # Check for extreme values
                param_min = np.min(param_data[np.isfinite(param_data)]) if np.any(np.isfinite(param_data)) else 0
                param_max = np.max(param_data[np.isfinite(param_data)]) if np.any(np.isfinite(param_data)) else 0
                param_mean = np.mean(param_data[np.isfinite(param_data)]) if np.any(np.isfinite(param_data)) else 0
                param_std = np.std(param_data[np.isfinite(param_data)]) if np.any(np.isfinite(param_data)) else 0
                
                # Add to table (only show first 20 parameters to avoid clutter)
                if len(table.rows) < 20:
                    table.add_row(
                        name,
                        str(param.shape),
                        f"{param_min:.6f}",
                        f"{param_max:.6f}",
                        f"{param_mean:.6f}",
                        f"{param_std:.6f}",
                        str(nan_count),
                        str(inf_count),
                        f"{zero_percent:.1f}%"
                    )
        
        console.print(table)
        
        # Summary
        console.print(f"\n[bold green]Summary:[/bold green]")
        console.print(f"Total DoRA parameters: {total_params:,}")
        console.print(f"Total NaN values: {total_nan}")
        console.print(f"Total Inf values: {total_inf}")
        
        if total_nan > 0 or total_inf > 0:
            console.print(f"[red]⚠️  CRITICAL: Found {total_nan} NaN and {total_inf} Inf values![/red]")
            console.print("[red]This is likely the cause of black image generation.[/red]")
            console.print("\nProblematic parameters:")
            for param_name in problematic_params[:10]:  # Show first 10
                console.print(f"  - {param_name}")
            return False
        else:
            console.print("[green]✓ No NaN or Inf values found in DoRA weights[/green]")
        
        # Check weight magnitudes
        all_weights = []
        for name, param in dora_unet.named_parameters():
            if 'lora_' in name or 'dora_' in name:
                param_data = param.detach().cpu().numpy()
                all_weights.extend(param_data.flatten())
        
        all_weights = np.array(all_weights)
        all_weights = all_weights[np.isfinite(all_weights)]  # Remove NaN/Inf
        
        if len(all_weights) > 0:
            weight_percentiles = np.percentile(all_weights, [1, 5, 25, 50, 75, 95, 99])
            console.print(f"\nWeight distribution percentiles:")
            console.print(f"  1%: {weight_percentiles[0]:.6f}")
            console.print(f"  5%: {weight_percentiles[1]:.6f}")
            console.print(f" 25%: {weight_percentiles[2]:.6f}")
            console.print(f" 50%: {weight_percentiles[3]:.6f}")
            console.print(f" 75%: {weight_percentiles[4]:.6f}")
            console.print(f" 95%: {weight_percentiles[5]:.6f}")
            console.print(f" 99%: {weight_percentiles[6]:.6f}")
            
            # Check for extreme weights
            if np.abs(weight_percentiles[0]) > 1.0 or np.abs(weight_percentiles[6]) > 1.0:
                console.print("[yellow]⚠️  Warning: Some weights are quite large (>1.0)[/yellow]")
                console.print("[yellow]This could cause numerical instability during generation[/yellow]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]Error analyzing weights: {e}[/red]")
        return False

def test_generation_with_base_model():
    """Test generation with base model to ensure it works."""
    console.print(Panel("[bold blue]Testing Base Model Generation[/bold blue]"))
    
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
        
        # Force VAE to float32
        pipeline.vae = pipeline.vae.to(torch.float32)
        
        console.print("Generating test image with base model...")
        
        result = pipeline(
            "a red apple on a white background",
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512,
            generator=torch.Generator(device=pipeline.device).manual_seed(42)
        )
        
        image = result.images[0]
        
        # Analyze image
        img_array = np.array(image)
        img_mean = img_array.mean()
        img_std = img_array.std()
        
        console.print(f"Base model image stats:")
        console.print(f"  Mean: {img_mean:.2f}")
        console.print(f"  Std: {img_std:.2f}")
        console.print(f"  Min: {img_array.min()}")
        console.print(f"  Max: {img_array.max()}")
        
        if img_mean > 10 and img_std > 5:
            console.print("[green]✓ Base model generation successful[/green]")
            image.save("base_model_test.png")
            console.print("Saved test image as 'base_model_test.png'")
            return True
        else:
            console.print("[red]✗ Base model generated poor quality image[/red]")
            return False
            
    except Exception as e:
        console.print(f"[red]Error testing base model: {e}[/red]")
        return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug DoRA weights for black image issues")
    parser.add_argument("--weights_path", type=str, required=True,
                       help="Path to DoRA weights directory")
    parser.add_argument("--test_base", action="store_true",
                       help="Test base model generation first")
    
    args = parser.parse_args()
    
    console.print(Panel("[bold green]DoRA Weights Debug Tool[/bold green]"))
    
    success = True
    
    if args.test_base:
        success = test_generation_with_base_model() and success
    
    success = analyze_dora_weights(args.weights_path) and success
    
    if success:
        console.print("\n[green]✓ Analysis complete - no critical issues found[/green]")
        console.print("[green]If you're still getting black images, the issue may be elsewhere.[/green]")
    else:
        console.print("\n[red]✗ Critical issues found in DoRA weights[/red]")
        console.print("[red]This is likely causing the black image generation.[/red]")
        console.print("\nSuggestions:")
        console.print("1. Retrain with lower learning rate")
        console.print("2. Use gradient clipping")
        console.print("3. Check your training data")
        console.print("4. Use more conservative DoRA settings (lower rank/alpha)")

if __name__ == "__main__":
    main()
