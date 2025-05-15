"""Example script demonstrating how to compile specific components of GaudiStableDiffusionXLPipeline."""

import torch
from diffusers import StableDiffusionXLPipeline
from optimum.habana.diffusers import (
    GaudiStableDiffusionXLPipeline,
    GaudiEulerDiscreteScheduler,
)
import sys
import os

# Add the parent directory to the Python path to import utility_diagnostic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility_diagnostic.utils.module_utils import (
    apply_compile_to_path,
    apply_compile_except,
    list_submodules,
    is_wrappable_module
)

def create_pipeline(model_name="stabilityai/stable-diffusion-xl-base-1.0", device="hpu", gaudi_config="Habana/stable-diffusion"):
    """Create and configure a GaudiStableDiffusionXLPipeline."""
    scheduler = GaudiEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
    pipeline = GaudiStableDiffusionXLPipeline.from_pretrained(
        model_name,
        scheduler=scheduler,
        use_habana=True,
        use_hpu_graphs=False,
        gaudi_config=gaudi_config,
        torch_dtype=torch.bfloat16
    )
    pipeline.set_progress_bar_config(disable=True)
    return pipeline.to(device)

def main():
    # Create pipeline
    pipeline = create_pipeline()

    # List all submodules to see what we can compile
    print("\nPipeline submodules:")
    for line in list_submodules(pipeline):
        print(line)

    # Example 1: Compile specific component (e.g., VAE decoder)
    print("\nExample 1: Compiling VAE decoder")
    try:
        # Compile only the VAE decoder
        pipeline = apply_compile_to_path(pipeline, "vae.decoder")
        print("Successfully compiled VAE decoder")
    except Exception as e:
        print(f"Failed to compile VAE decoder: {str(e)}")

    # Example 2: Compile all except specific components
    print("\nExample 2: Compiling all except specific components")
    try:
        # Compile everything except the text encoder and tokenizer
        exclude_paths = ["text_encoder", "tokenizer"]
        pipeline = apply_compile_except(pipeline, exclude_paths)
        print(f"Successfully compiled all components except {exclude_paths}")
    except Exception as e:
        print(f"Failed to compile components: {str(e)}")

    # Example 3: Compile specific submodules within components
    print("\nExample 3: Compiling specific submodules")
    try:
        # Compile specific attention layers in the UNet
        target_paths = [
            "unet.down_blocks.0.attentions.0",
            "unet.up_blocks.1.attentions.1"
        ]
        for path in target_paths:
            pipeline = apply_compile_to_path(pipeline, path)
            print(f"Successfully compiled {path}")
    except Exception as e:
        print(f"Failed to compile submodules: {str(e)}")

    # Example 4: Test the compiled pipeline
    print("\nExample 4: Testing the compiled pipeline")
    try:
        with torch.no_grad():
            output = pipeline(
                prompt="A beautiful sunset over mountains",
                num_inference_steps=30
            )
            if hasattr(output, 'images'):
                image = output.images[0]
                image.save("test_output.png")
                print("Successfully generated and saved test image")
            else:
                print("Pipeline output format unexpected")
    except Exception as e:
        print(f"Failed to generate image: {str(e)}")

if __name__ == "__main__":
    main()