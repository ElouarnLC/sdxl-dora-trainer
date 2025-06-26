#!/usr/bin/env python3
"""
Fix DoRA weights by removing NaN/Inf values and clipping extreme weights.
"""

import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file, save_file
from rich.console import Console
from rich.panel import Panel

console = Console()

def fix_dora_weights(weights_path: str, output_path: str = None, clip_value: float = 1.0):
    """Fix DoRA weights by removing NaN/Inf and clipping extreme values."""
    console.print(Panel("[bold blue]Fixing DoRA Weights[/bold blue]"))
    
    weights_path = Path(weights_path)
    if not weights_path.exists():
        console.print(f"[red]Error: Weights path does not exist: {weights_path}[/red]")
        return False
    
    if output_path is None:
        output_path = weights_path.parent / f"{weights_path.name}_fixed"
    else:
        output_path = Path(output_path)
    
    try:
        # Find adapter weights file
        adapter_weights_file = None
        for file_path in weights_path.rglob("*.safetensors"):
            if "adapter" in file_path.name:
                adapter_weights_file = file_path
                break
        
        if not adapter_weights_file:
            console.print("[red]No adapter weights file found (.safetensors)[/red]")
            return False
        
        console.print(f"Loading weights from: {adapter_weights_file}")
        
        # Load weights
        weights = load_file(adapter_weights_file)
        
        fixed_weights = {}
        total_nan_fixed = 0
        total_inf_fixed = 0
        total_clipped = 0
        
        console.print("Processing weights...")
        
        for name, tensor in weights.items():
            # Convert to float32 for processing
            tensor_data = tensor.float()
            
            # Count issues
            nan_count = torch.isnan(tensor_data).sum().item()
            inf_count = torch.isinf(tensor_data).sum().item()
            
            # Fix NaN values (replace with zeros)
            if nan_count > 0:
                console.print(f"  Fixing {nan_count} NaN values in {name}")
                tensor_data = torch.where(torch.isnan(tensor_data), torch.zeros_like(tensor_data), tensor_data)
                total_nan_fixed += nan_count
            
            # Fix Inf values (replace with clipped values)
            if inf_count > 0:
                console.print(f"  Fixing {inf_count} Inf values in {name}")
                tensor_data = torch.where(torch.isinf(tensor_data), 
                                        torch.sign(tensor_data) * clip_value, 
                                        tensor_data)
                total_inf_fixed += inf_count
            
            # Clip extreme values
            original_tensor = tensor_data.clone()
            tensor_data = torch.clamp(tensor_data, -clip_value, clip_value)
            clipped_count = (tensor_data != original_tensor).sum().item()
            if clipped_count > 0:
                console.print(f"  Clipped {clipped_count} extreme values in {name}")
                total_clipped += clipped_count
            
            # Convert back to original dtype
            fixed_weights[name] = tensor_data.to(tensor.dtype)
        
        # Save fixed weights
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / "adapter_model.safetensors"
        
        console.print(f"Saving fixed weights to: {output_file}")
        save_file(fixed_weights, output_file)
        
        # Copy other files
        for file_path in weights_path.iterdir():
            if file_path.is_file() and file_path.name != "adapter_model.safetensors":
                import shutil
                shutil.copy2(file_path, output_path / file_path.name)
        
        console.print(f"\n[green]✓ Fixed weights saved to {output_path}[/green]")
        console.print(f"  NaN values fixed: {total_nan_fixed}")
        console.print(f"  Inf values fixed: {total_inf_fixed}")
        console.print(f"  Values clipped: {total_clipped}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]Error fixing weights: {e}[/red]")
        return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix DoRA weights with NaN/Inf values")
    parser.add_argument("--weights_path", type=str, required=True,
                       help="Path to DoRA weights directory")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output path for fixed weights (default: input_path_fixed)")
    parser.add_argument("--clip_value", type=float, default=1.0,
                       help="Maximum absolute value for weight clipping (default: 1.0)")
    
    args = parser.parse_args()
    
    console.print(Panel("[bold green]DoRA Weights Fix Tool[/bold green]"))
    
    success = fix_dora_weights(args.weights_path, args.output_path, args.clip_value)
    
    if success:
        console.print("\n[green]✓ Weights successfully fixed![/green]")
        console.print("Try using the fixed weights for inference.")
    else:
        console.print("\n[red]✗ Failed to fix weights[/red]")

if __name__ == "__main__":
    main()
