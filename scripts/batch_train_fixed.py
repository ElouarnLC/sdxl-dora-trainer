#!/usr/bin/env python3
"""
Batch processing script for SDXL DoRA training.
Allows training multiple models with different configurations.
"""

import os
import sys
import json
import yaml
import time
import argparse
from pathlib import Path
from typing import Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from sdxl_dora_trainer import DoRATrainer, TrainingConfig

console = Console()

class BatchTrainer:
    """Manages batch training of multiple models."""
    
    def __init__(self, batch_config_path: str):
        self.batch_config_path = Path(batch_config_path)
        self.batch_config = self._load_batch_config()
        self.results = []
    
    def _load_batch_config(self) -> Dict:
        """Load batch configuration file."""
        if not self.batch_config_path.exists():
            raise FileNotFoundError(f"Batch config not found: {self.batch_config_path}")
        
        if self.batch_config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(self.batch_config_path, 'r') as f:
                return yaml.safe_load(f)
        elif self.batch_config_path.suffix.lower() == '.json':
            with open(self.batch_config_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {self.batch_config_path.suffix}")
    
    def _create_training_config(self, job_config: Dict) -> TrainingConfig:
        """Create training configuration from job config."""
        # Start with base config
        base_config = self.batch_config.get('base_config', {})
        
        # Merge with job-specific config
        merged_config = {**base_config, **job_config}
        
        # Validate mixed_precision setting for stability
        if merged_config.get('mixed_precision') == 'fp16':
            console.print(f"[yellow]Warning: Job '{job_config.get('name', 'unknown')}' uses fp16 mixed precision[/yellow]")
            console.print("[yellow]This may cause black images with DoRA training. Consider using 'no' or 'bf16'[/yellow]")
        
        # Create TrainingConfig object
        config = TrainingConfig()
        for key, value in merged_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def run_batch(self):
        """Run all training jobs in the batch."""
        jobs = self.batch_config.get('jobs', [])
        
        if not jobs:
            console.print("[red]No jobs found in batch configuration[/red]")
            return
        
        console.print(Panel(f"[bold green]Starting Batch Training[/bold green]\n{len(jobs)} jobs to process"))
        
        for i, job in enumerate(jobs, 1):
            job_name = job.get('name', f'Job_{i}')
            console.print(f"\n[bold cyan]Starting Job {i}/{len(jobs)}: {job_name}[/bold cyan]")
            
            try:
                # Create training configuration
                config = self._create_training_config(job)
                
                # Ensure unique output directory
                if not config.output_dir.endswith(job_name):
                    config.output_dir = os.path.join(config.output_dir, job_name)
                
                # Create trainer and run
                trainer = DoRATrainer(config)
                start_time = time.time()
                
                trainer.train()
                
                end_time = time.time()
                duration = end_time - start_time
                
                self.results.append({
                    'job_name': job_name,
                    'status': 'success',
                    'duration': duration,
                    'output_dir': config.output_dir
                })
                
                console.print(f"[green]✓[/green] Job {job_name} completed successfully")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Batch training interrupted by user[/yellow]")
                self.results.append({
                    'job_name': job_name,
                    'status': 'interrupted',
                    'duration': time.time() - start_time if 'start_time' in locals() else 0
                })
                break
                
            except Exception as e:
                console.print(f"[red]✗[/red] Job {job_name} failed: {e}")
                self.results.append({
                    'job_name': job_name,
                    'status': 'failed',
                    'error': str(e),
                    'duration': time.time() - start_time if 'start_time' in locals() else 0
                })
                
                # Continue with next job or stop based on configuration
                if not self.batch_config.get('continue_on_error', True):
                    console.print("[red]Stopping batch training due to error[/red]")
                    break
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print batch training summary."""
        console.print(Panel("[bold green]Batch Training Summary[/bold green]"))
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Job Name", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Output Directory", style="blue")
        
        for result in self.results:
            status = result['status']
            if status == 'success':
                status_color = "[green]✓ Success[/green]"
            elif status == 'failed':
                status_color = "[red]✗ Failed[/red]"
            else:
                status_color = "[yellow]⚠ Interrupted[/yellow]"
            
            duration_str = f"{result['duration']:.1f}s"
            if result['duration'] > 3600:
                duration_str = f"{result['duration']/3600:.1f}h"
            elif result['duration'] > 60:
                duration_str = f"{result['duration']/60:.1f}m"
            
            table.add_row(
                result['job_name'],
                status_color,
                duration_str,
                result.get('output_dir', 'N/A')
            )
        
        console.print(table)
        
        # Print statistics
        total_jobs = len(self.results)
        successful_jobs = sum(1 for r in self.results if r['status'] == 'success')
        failed_jobs = sum(1 for r in self.results if r['status'] == 'failed')
        
        console.print(f"\n[bold]Results:[/bold] {successful_jobs}/{total_jobs} successful, {failed_jobs} failed")

def create_example_batch_config():
    """Create an example batch configuration file."""
    example_config = {
        "base_config": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "output_dir": "./batch_output",
            "cache_dir": "./cache",
            "max_train_steps": 1000,
            "resolution": 1024,
            "mixed_precision": "no",  # Changed from fp16 to avoid black image issues
            "gradient_checkpointing": True,
            "use_8bit_adam": True,
            "report_to": "tensorboard"
        },
        "continue_on_error": True,
        "jobs": [
            {
                "name": "portraits_rank32",
                "dataset_path": "./datasets/portraits",
                "rank": 32,
                "alpha": 16,
                "learning_rate": 1e-4,
                "project_name": "portraits-rank32"
            },
            {
                "name": "portraits_rank64",
                "dataset_path": "./datasets/portraits",
                "rank": 64,
                "alpha": 32,
                "learning_rate": 1e-4,
                "project_name": "portraits-rank64"
            },
            {
                "name": "landscapes_rank64",
                "dataset_path": "./datasets/landscapes",
                "rank": 64,
                "alpha": 32,
                "learning_rate": 5e-5,
                "project_name": "landscapes-rank64"
            }
        ]
    }
    
    with open("batch_config.yaml", "w") as f:
        yaml.dump(example_config, f, default_flow_style=False, indent=2)
    
    console.print("[green]✓[/green] Example batch configuration created: batch_config.yaml")

def main():
    """Main entry point for batch training."""
    parser = argparse.ArgumentParser(description="Batch training for SDXL DoRA models")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to batch configuration file")
    parser.add_argument("--create-example", action="store_true",
                       help="Create an example batch configuration file")
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_batch_config()
        return
    
    if not args.config:
        console.print("[red]Error: --config is required[/red]")
        parser.print_help()
        sys.exit(1)
    
    try:
        batch_trainer = BatchTrainer(args.config)
        batch_trainer.run_batch()
    except KeyboardInterrupt:
        console.print("\n[yellow]Batch training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Batch training failed: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
