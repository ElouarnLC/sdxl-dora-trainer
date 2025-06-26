# Example Inference Script
# This script shows how to generate images with trained DoRA models

from pathlib import Path
from PIL import Image
from inference import SDXLDoRAInference

def main():
    """Example inference script."""
    
    # Configuration
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    dora_weights = "./output/checkpoints/checkpoint-1000"  # Path to your trained weights
    output_dir = "./generated_images"
    
    # Check if weights exist
    if not Path(dora_weights).exists():
        print(f"Error: DoRA weights not found at {dora_weights}")
        print("Train a model first or check the path.")
        return
    
    # Initialize inference
    print("Loading model...")
    inference = SDXLDoRAInference(
        base_model_path=base_model,
        dora_weights_path=dora_weights,
        device="auto"  # Automatically choose best device
    )
    
    # Load the model
    inference.load_model()
    
    # Define prompts
    prompts = [
        "a beautiful sunset over mountains",
        "a portrait of a wise old wizard",
        "a futuristic city with flying cars",
        "a cute cat playing in a garden",
        "an artistic painting of flowers"
    ]
    
    # Generation settings
    generation_params = {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "width": 1024,
        "height": 1024,
        "seed": 42,  # For reproducible results
        "output_dir": output_dir
    }
    
    # Generate images
    print(f"Generating {len(prompts)} images...")
    try:
        images = inference.generate_images(
            prompts=prompts,
            **generation_params
        )
        
        print(f"Successfully generated {len(images)} images!")
        print(f"Images saved to: {output_dir}")
        
    except Exception as e:
        print(f"Generation failed: {e}")
        raise

def interactive_generation():
    """Example of interactive generation."""
    
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    dora_weights = "./output/checkpoints/checkpoint-1000"
    
    # Initialize inference
    inference = SDXLDoRAInference(
        base_model_path=base_model,
        dora_weights_path=dora_weights
    )
    
    inference.load_model()
    
    # Run interactive mode
    inference.interactive_mode()

if __name__ == "__main__":
    # Run batch generation
    main()
    
    # Uncomment to try interactive mode
    # interactive_generation()
