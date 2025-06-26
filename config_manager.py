#!/usr/bin/env python3
"""
Configuration management for SDXL DoRA Trainer.
Supports YAML and JSON configuration files.
"""

import json
import yaml
from pathlib import Path
from dataclasses import asdict
from sdxl_dora_trainer import TrainingConfig

class ConfigManager:
    """Manages configuration loading and saving."""
    
    @staticmethod
    def load_config(config_path: str) -> TrainingConfig:
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Determine file format
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Create config object
        config = TrainingConfig()
        
        # Update config with loaded values
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown config parameter: {key}")
        
        return config
    
    @staticmethod
    def save_config(config: TrainingConfig, config_path: str, format: str = 'yaml'):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(config)
        
        if format.lower() in ['yaml', 'yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def create_default_config(output_path: str):
        """Create a default configuration file."""
        config = TrainingConfig()
        ConfigManager.save_config(config, output_path)
        print(f"Default configuration saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration management tool")
    parser.add_argument("action", choices=["create", "validate"], 
                       help="Action to perform")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--format", type=str, default="yaml", 
                       choices=["yaml", "json"],
                       help="Configuration file format")
    
    args = parser.parse_args()
    
    if args.action == "create":
        ConfigManager.create_default_config(args.config)
    elif args.action == "validate":
        try:
            config = ConfigManager.load_config(args.config)
            print("Configuration is valid!")
        except Exception as e:
            print(f"Configuration error: {e}")
