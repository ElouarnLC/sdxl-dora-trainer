#!/usr/bin/env python3
"""
Model inference script for SDXL DoRA trained models.
Allows generation of images using your fine-tuned models.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class SDXLDoRAInference:
    """Inference class for SDXL DoRA models."""
    
    def __init__(self, 
                 base_model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 dora_weights_path: Optional[str] = None,
                 device: str = "auto"):
        
        self.base_model_path = base_model_path
        self.dora_weights_path = dora_weights_path
        self.device = self._setup_device(device)
        self.pipeline = None
        
        console.print(f"[cyan]Device: {self.device}[/cyan]")
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the SDXL model with DoRA weights."""
        console.print(Panel("[bold blue]Loading Model[/bold blue]"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Load base pipeline
            task = progress.add_task("Loading base SDXL model...", total=None)
              # Load with better error handling and precision settings
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float32,  # Always use float32 for stability
                use_safetensors=True,
                variant=None,  # Don't use fp16 variant
                safety_checker=None,
                requires_safety_checker=False
            )
            
            progress.update(task, description="Base model loaded")
              # Load DoRA weights if provided
            if self.dora_weights_path:
                task = progress.add_task("Loading DoRA weights...", total=None)
                
                try:
                    # Load DoRA weights with error checking
                    console.print(f"[cyan]Loading DoRA weights from: {self.dora_weights_path}[/cyan]")
                    
                    # Check if the path exists and contains valid weights
                    weights_path = Path(self.dora_weights_path)
                    if not weights_path.exists():
                        raise FileNotFoundError(f"DoRA weights path does not exist: {self.dora_weights_path}")
                    
                    # Check for required files
                    adapter_config = weights_path / "adapter_config.json"
                    adapter_weights = weights_path / "adapter_model.safetensors"
                    
                    if not adapter_config.exists():
                        raise FileNotFoundError(f"adapter_config.json not found in {self.dora_weights_path}")
                    if not adapter_weights.exists():
                        raise FileNotFoundError(f"adapter_model.safetensors not found in {self.dora_weights_path}")
                    
                    # Load DoRA weights to UNet
                    self.pipeline.unet = PeftModel.from_pretrained(
                        self.pipeline.unet,
                        self.dora_weights_path,
                        torch_dtype=torch.float32  # Always use float32 for stability
                    )
                    
                    # Verify weights don't contain NaN or inf values
                    nan_count = 0
                    inf_count = 0
                    total_params = 0
                    
                    for name, param in self.pipeline.unet.named_parameters():
                        total_params += param.numel()
                        if torch.isnan(param).any():
                            nan_count += torch.isnan(param).sum().item()
                            console.print(f"[yellow]Warning: NaN values in parameter {name}[/yellow]")
                        if torch.isinf(param).any():
                            inf_count += torch.isinf(param).sum().item()
                            console.print(f"[yellow]Warning: Inf values in parameter {name}[/yellow]")
                    
                    if nan_count > 0 or inf_count > 0:
                        console.print(f"[red]Warning: DoRA weights contain {nan_count} NaN and {inf_count} Inf values out of {total_params} parameters[/red]")
                        console.print("[red]This may cause black/corrupted image generation[/red]")
                        console.print("[cyan]Consider using the fix_dora_weights.py script to repair the weights[/cyan]")
                        
                        # Option to continue with a warning
                        response = input("Continue anyway? (y/N): ").strip().lower()
                        if response != 'y':
                            raise ValueError("DoRA weights contain invalid values")
                    else:
                        console.print(f"[green]✓[/green] DoRA weights validated ({total_params:,} parameters)")
                    
                    progress.update(task, description="DoRA weights loaded and validated")
                    
                except Exception as e:
                    console.print(f"[red]Failed to load DoRA weights: {e}[/red]")
                    console.print("[yellow]Continuing with base model only[/yellow]")
                    # Reset pipeline to base model if DoRA loading failed
                    self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                        self.base_model_path,
                        torch_dtype=torch.float32,
                        use_safetensors=True,
                        variant=None,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                    
            else:
                console.print("[yellow]No DoRA weights specified, using base model only[/yellow]")
            
            # Move to device
            task = progress.add_task("Moving to device...", total=None)
            self.pipeline = self.pipeline.to(self.device)
            
            # Force VAE to use float32 to prevent NaN issues
            self.pipeline.vae = self.pipeline.vae.to(torch.float32)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline.unet, "enable_xformers_memory_efficient_attention"):
                try:
                    self.pipeline.unet.enable_xformers_memory_efficient_attention()
                    progress.update(task, description="XFormers enabled")
                except Exception:
                    pass
            
            progress.update(task, description="Model ready!")
        
        console.print("[green]✓[/green] Model loaded successfully")
    
    def generate_images(self,
                       prompts: List[str],
                       negative_prompt: str = "",
                       num_images_per_prompt: int = 1,
                       num_inference_steps: int = 50,
                       guidance_scale: float = 7.5,
                       width: int = 1024,
                       height: int = 1024,
                       seed: Optional[int] = None,
                       output_dir: str = "./generated_images") -> List[Image.Image]:
        """Generate images from prompts."""
        
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
          # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
        
        all_images = []
        
        console.print(Panel(f"[bold green]Generating Images[/bold green]\n{len(prompts)} prompts"))
        
        with Progress(console=console) as progress:
            task = progress.add_task("Generating...", total=len(prompts))
            
            for i, prompt in enumerate(prompts):
                console.print(f"\n[cyan]Prompt {i+1}:[/cyan] {prompt}")
                
                try:
                    # Generate images with conservative settings to avoid black images
                    # Don't use autocast to prevent precision issues
                    
                    # Check model parameters before generation
                    has_invalid = False
                    for name, param in self.pipeline.unet.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            console.print(f"[red]Warning: Invalid values in {name}[/red]")
                            has_invalid = True
                    
                    if has_invalid:
                        console.print("[yellow]Proceeding with generation despite invalid parameters[/yellow]")
                        console.print("[yellow]Consider using the fix_dora_weights.py script[/yellow]")
                    
                    # Generate with stable settings
                    result = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        generator=torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
                    )
                    
                    images = result.images
                    
                    # Check if images are valid (not all black)
                    for j, image in enumerate(images):
                        import numpy as np
                        img_array = np.array(image)
                        img_mean = img_array.mean()
                        img_std = img_array.std()
                        
                        if img_mean < 10 and img_std < 5:
                            console.print(f"[yellow]Warning: Image {j+1} appears to be mostly black (mean: {img_mean:.2f}, std: {img_std:.2f})[/yellow]")
                            console.print("[yellow]This may indicate an issue with the DoRA weights[/yellow]")
                    
                    all_images.extend(images)
                    
                    # Save images
                    for j, image in enumerate(images):
                        filename = f"prompt_{i+1:03d}_image_{j+1:02d}.png"
                        image_path = output_path / filename
                        image.save(image_path)
                        console.print(f"[green]✓[/green] Saved: {filename}")
                    
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    console.print(f"[red]✗[/red] Failed to generate image for prompt {i+1}: {e}")
                    progress.update(task, advance=1)
                    continue
        
        console.print(f"\n[green]✓[/green] Generated {len(all_images)} images in {output_dir}")
        return all_images
    
    def interactive_mode(self):
        """Interactive generation mode."""
        console.print(Panel("[bold green]Interactive Mode[/bold green]\nType 'quit' to exit"))
        
        while True:
            try:
                prompt = input("\nEnter prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    continue
                
                # Optional parameters
                try:
                    steps = input("Inference steps (default 50): ").strip()
                    steps = int(steps) if steps else 50
                    
                    guidance = input("Guidance scale (default 7.5): ").strip()
                    guidance = float(guidance) if guidance else 7.5
                    
                    seed_input = input("Seed (optional): ").strip()
                    seed = int(seed_input) if seed_input else None
                    
                except ValueError:
                    console.print("[yellow]Invalid input, using defaults[/yellow]")
                    steps, guidance, seed = 50, 7.5, None
                
                # Generate
                self.generate_images(
                    prompts=[prompt],
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    seed=seed
                )
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Exiting interactive mode[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SDXL DoRA Model Inference")
    
    # Model parameters
    parser.add_argument("--base_model", type=str, 
                       default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="Base SDXL model path")
    parser.add_argument("--dora_weights", type=str, default=None,
                       help="Path to DoRA weights")
    parser.add_argument("--validate_weights", action="store_true",
                       help="Validate DoRA weights and exit (don't generate images)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu", "mps"],
                       help="Device to use")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, action="append",
                       help="Prompt(s) to generate (can be used multiple times)")
    parser.add_argument("--prompts_file", type=str,
                       help="File containing prompts (one per line)")
    parser.add_argument("--negative_prompt", type=str, default="",
                       help="Negative prompt")
    parser.add_argument("--num_images", type=int, default=1,
                       help="Number of images per prompt")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--width", type=int, default=1024,
                       help="Image width")
    parser.add_argument("--height", type=int, default=1024,
                       help="Image height")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./generated_images",
                       help="Output directory")
    
    # Mode
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.prompt and not args.prompts_file:
        console.print("[red]Error: Either --prompt, --prompts_file, or --interactive must be specified[/red]")
        parser.print_help()
        sys.exit(1)
      # Create inference object
    inference = SDXLDoRAInference(
        base_model_path=args.base_model,
        dora_weights_path=args.dora_weights,
        device=args.device
    )
    
    # Handle weight validation mode
    if args.validate_weights:
        if not args.dora_weights:
            console.print("[red]Error: --dora_weights required for validation[/red]")
            sys.exit(1)
        
        console.print(Panel("[bold blue]DoRA Weight Validation Mode[/bold blue]"))
        
        try:
            # Load base model first
            console.print("[cyan]Loading base model for validation...[/cyan]")
            inference.pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.base_model,
                torch_dtype=torch.float32,
                use_safetensors=True,
                variant=None,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Try to load DoRA weights
            console.print(f"[cyan]Validating DoRA weights: {args.dora_weights}[/cyan]")
            weights_path = Path(args.dora_weights)
            
            if not weights_path.exists():
                console.print(f"[red]Error: DoRA weights path does not exist: {args.dora_weights}[/red]")
                sys.exit(1)
            
            # Check for required files
            adapter_config = weights_path / "adapter_config.json"
            adapter_weights = weights_path / "adapter_model.safetensors"
            
            if not adapter_config.exists():
                console.print(f"[red]Error: adapter_config.json not found in {args.dora_weights}[/red]")
                sys.exit(1)
            if not adapter_weights.exists():
                console.print(f"[red]Error: adapter_model.safetensors not found in {args.dora_weights}[/red]")
                sys.exit(1)
            
            # Load and validate weights
            from peft import PeftModel
            unet_with_dora = PeftModel.from_pretrained(
                inference.pipeline.unet,
                args.dora_weights,
                torch_dtype=torch.float32
            )
            
            # Check for NaN/Inf values
            nan_count = 0
            inf_count = 0
            total_params = 0
            
            for name, param in unet_with_dora.named_parameters():
                total_params += param.numel()
                if torch.isnan(param).any():
                    nan_count += torch.isnan(param).sum().item()
                if torch.isinf(param).any():
                    inf_count += torch.isinf(param).sum().item()
            
            if nan_count > 0 or inf_count > 0:
                console.print(f"[red]VALIDATION FAILED: DoRA weights contain {nan_count} NaN and {inf_count} Inf values[/red]")
                console.print("[yellow]Consider using scripts/fix_dora_weights.py to repair the weights[/yellow]")
                sys.exit(1)
            else:
                console.print(f"[green]VALIDATION PASSED: DoRA weights are valid ({total_params:,} parameters)[/green]")
                sys.exit(0)
                
        except Exception as e:
            console.print(f"[red]VALIDATION FAILED: {e}[/red]")
            sys.exit(1)
    
    # Load model for normal inference
    try:
        inference.load_model()
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        sys.exit(1)
    
    # Run inference
    try:
        if args.interactive:
            inference.interactive_mode()
        else:
            # Collect prompts
            prompts = []
            
            if args.prompt:
                prompts.extend(args.prompt)
            
            if args.prompts_file:
                prompts_file = Path(args.prompts_file)
                if prompts_file.exists():
                    with open(prompts_file, 'r', encoding='utf-8') as f:
                        file_prompts = [line.strip() for line in f if line.strip()]
                        prompts.extend(file_prompts)
                else:
                    console.print(f"[red]Prompts file not found: {args.prompts_file}[/red]")
                    sys.exit(1)
            
            if not prompts:
                console.print("[red]No prompts provided[/red]")
                sys.exit(1)
            
            # Generate images
            inference.generate_images(
                prompts=prompts,
                negative_prompt=args.negative_prompt,
                num_images_per_prompt=args.num_images,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                width=args.width,
                height=args.height,
                seed=args.seed,
                output_dir=args.output_dir
            )
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Generation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Generation failed: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
