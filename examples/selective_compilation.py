"""Example script demonstrating selective compilation disabling for GaudiStableDiffusionXLPipeline.

This script shows how to:
1. Create a pipeline with selective compilation disabling
2. Run inference with different compilation configurations
3. Save the generated images
"""

import torch
from diffusers import StableDiffusionXLPipeline
from optimum.habana.diffusers import (
    GaudiStableDiffusionXLPipeline,
    GaudiEulerDiscreteScheduler
)
from typing import List, Union, Optional, Tuple

def get_module_at_path(module: torch.nn.Module, path: str) -> Tuple[torch.nn.Module, str]:
    """
    Get a module at the specified path and its parent attribute name.

    Args:
        module: The root module to search in
        path: The dot-separated path to the module

    Returns:
        Tuple of (module, parent_attribute_name)
    """
    parts = path.split('.')
    current = module
    parent = None
    parent_attr = None

    for i, part in enumerate(parts):
        try:
            parent = current
            parent_attr = part
            current = getattr(current, part)
        except AttributeError as e:
            raise AttributeError(f"Could not find module at path '{path}': {str(e)}")

    return current, parent_attr

def create_pipeline_with_selective_compilation(
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device: str = "hpu",
    gaudi_config: str = "Habana/stable-diffusion",
    disable_compilation_for: Optional[Union[str, List[str]]] = None
) -> GaudiStableDiffusionXLPipeline:
    """
    Create a GaudiStableDiffusionXLPipeline with selective compilation disabling.

    Args:
        model_name (str): The model name or path
        device (str): The device to run on ('hpu', 'cuda', 'cpu')
        gaudi_config (str): The Gaudi configuration to use. Defaults to "Habana/stable-diffusion"
        disable_compilation_for (Union[str, List[str]], optional):
            Submodule paths to disable compilation for. Can be:
            - Top-level components: 'text_encoder', 'vae', 'unet', 'scheduler'
            - Nested components: 'vae.decoder', 'vae.decoder.up_block.0', etc.
            - Or any combination of these
    """
    # Create scheduler and pipeline
    scheduler = GaudiEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
    pipeline = GaudiStableDiffusionXLPipeline.from_pretrained(
        model_name,
        scheduler=scheduler,
        use_habana=True,
        use_hpu_graphs=False,
        gaudi_config=gaudi_config,
        torch_dtype=torch.bfloat16
    )

    # Move pipeline to device first
    pipeline = pipeline.to(device)

    # Convert single string to list for uniform handling
    if isinstance(disable_compilation_for, str):
        disable_compilation_for = [disable_compilation_for]

    # Store the disabled components for later use
    pipeline._disabled_components = set(disable_compilation_for) if disable_compilation_for else set()

    # First, handle components that should not be compiled
    if pipeline._disabled_components:
        for component_path in pipeline._disabled_components:
            try:
                # Get the module and its parent attribute name
                module, parent_attr = get_module_at_path(pipeline, component_path)

                # Get the parent module
                if '.' in component_path:
                    parent_path = '.'.join(component_path.split('.')[:-1])
                    parent, _ = get_module_at_path(pipeline, parent_path)
                else:
                    parent = pipeline

                # Disable compilation for this module
                disabled_module = torch.compiler.disable(module)
                setattr(parent, parent_attr, disabled_module)
            except Exception as e:
                print(f"Warning: Could not disable compilation for {component_path}: {str(e)}")

    # Then compile the remaining components
    try:
        # Create a wrapper for the pipeline's forward method
        original_forward = pipeline.__call__

        def wrapped_forward(*args, **kwargs):
            # Store original components
            original_components = {}

            # Temporarily replace disabled components with their non-compiled versions
            for component_path in pipeline._disabled_components:
                try:
                    # Get the module and its parent
                    module, parent_attr = get_module_at_path(pipeline, component_path)
                    if '.' in component_path:
                        parent_path = '.'.join(component_path.split('.')[:-1])
                        parent, _ = get_module_at_path(pipeline, parent_path)
                    else:
                        parent = pipeline

                    # Store original component
                    original_components[component_path] = (parent, parent_attr, getattr(parent, parent_attr))

                    # Replace with non-compiled version
                    disabled_module = torch.compiler.disable(module)
                    setattr(parent, parent_attr, disabled_module)
                except Exception as e:
                    print(f"Warning: Could not disable compilation for {component_path}: {str(e)}")

            try:
                # Run the pipeline
                result = original_forward(*args, **kwargs)
            finally:
                # Restore original components
                for component_path, (parent, parent_attr, original) in original_components.items():
                    setattr(parent, parent_attr, original)

            return result

        # Replace the pipeline's forward method
        pipeline.__call__ = wrapped_forward

        # Compile the pipeline with HPU backend
        pipeline = torch.compile(pipeline, backend="hpu_backend")
    except Exception as e:
        print(f"Warning: Could not compile pipeline: {str(e)}")

    return pipeline

def run_inference(
    pipeline: GaudiStableDiffusionXLPipeline,
    prompt: str = "A beautiful sunset over mountains",
    num_inference_steps: int = 50
) -> torch.Tensor:
    """
    Run inference with the pipeline.

    Args:
        pipeline: The pipeline to use
        prompt: The text prompt
        num_inference_steps: Number of inference steps

    Returns:
        The generated image
    """
    return pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps
    ).images[0]

def main():
    """Main function demonstrating different compilation configurations."""
    # Example 1: Disable compilation for VAE decoder only
    print("Running with VAE decoder compilation disabled...")
    pipeline1 = create_pipeline_with_selective_compilation(
        disable_compilation_for="vae.decoder",
        gaudi_config="Habana/stable-diffusion"
    )
    image1 = run_inference(pipeline1)
    image1.save("image_vae_decoder_disabled.png")
    print("Saved image_vae_decoder_disabled.png")

    # Example 2: Disable compilation for multiple nested components
    print("\nRunning with multiple nested components compilation disabled...")
    pipeline2 = create_pipeline_with_selective_compilation(
        disable_compilation_for=["vae.decoder.up_block.0", "unet.mid_block"],
        gaudi_config="Habana/stable-diffusion"
    )
    image2 = run_inference(pipeline2)
    image2.save("image_nested_disabled.png")
    print("Saved image_nested_disabled.png")

    # Example 3: Run with full compilation
    print("\nRunning with full compilation...")
    pipeline3 = create_pipeline_with_selective_compilation(
        gaudi_config="Habana/stable-diffusion"
    )
    image3 = run_inference(pipeline3)
    image3.save("image_fully_compiled.png")
    print("Saved image_fully_compiled.png")

if __name__ == "__main__":
    main()