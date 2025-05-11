"""Diagnostic tool for diffusion models.

This script provides functionality to diagnose and test diffusion models,
particularly focusing on module compilation and performance. It supports
various devices (HPU, CUDA, CPU) and provides comprehensive logging and
monitoring capabilities.
"""

import os
import sys
import torch
import numpy as np
import copy
from typing import List, Dict, Any, Optional

# Get the package root directory
package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from utils import (
    # Module utilities
    get_all_submodules,
    filter_submodules,
    is_wrappable_module,
    apply_compile_to_path,
    apply_compile_except,
    get_submodule_type,
    get_submodule_orig_type,

    # Image utilities
    is_blank,
    save_image_tensor,

    # Pipeline utilities
    create_pipeline,
    save_results,
    print_test_summary,
    list_pipeline_submodules,

    # Logging utilities
    setup_logging,
    setup_tensorboard,
    setup_wandb,
    Timer,
    log_metrics,
    cleanup_logging,

    # Argument utilities
    parse_arguments,
    get_config_from_args,
    validate_args
)


def run_diagnostic(
    model_name: str,
    device: str,
    mode: str,
    filter_type: str,
    output_dir: str,
    exclude_path: Optional[str] = None,
    bad_paths_file: Optional[str] = None,
    gaudi_config: Optional[str] = None,
    logger: Optional[Any] = None,
    writer: Optional[Any] = None,
    wandb_run: Optional[Any] = None
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Run diagnostic tests on a diffusion model.

    Args:
        model_name (str): Name of the model to test.
        device (str): Device to run the model on.
        mode (str): Test mode ("single" or "compile_except").
        filter_type (str): Type of modules to test ("all", "leaf", or "non-leaf").
        output_dir (str): Directory to save test results.
        exclude_path (Optional[str]): Path to exclude from testing.
        bad_paths_file (Optional[str]): File containing paths to exclude.
        gaudi_config (Optional[str]): Path to Gaudi configuration file.
        logger (Optional[Any]): Logger instance.
        writer (Optional[Any]): TensorBoard writer.
        wandb_run (Optional[Any]): Weights & Biases run.

    Returns:
        tuple[List[Dict[str, Any]], List[str]]: Test results and bad paths.
    """
    # Initialize results and bad paths
    results = []
    bad_paths = []

    # Get excluded paths
    exclude_paths = []
    if exclude_path:
        exclude_paths.append(exclude_path)
    if bad_paths_file and os.path.exists(bad_paths_file):
        with open(bad_paths_file, 'r') as f:
            exclude_paths.extend([line.strip() for line in f])

    # Create pipeline
    init_pipeline = create_pipeline(model_name, device, gaudi_config)

    # Store original module states for resetting
    original_states = {}
    for path, module in get_all_submodules(init_pipeline):
        original_states[path] = {
            "state": copy.deepcopy(module.state_dict()),
            "type": type(module)
        }

    # Get all submodules
    all_modules = list(get_all_submodules(init_pipeline))

    # Filter modules
    filtered_modules = filter_submodules(all_modules, filter_type)

    # Create images subdirectory
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Cleanup init pipeline
    del init_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run tests
    with Timer("Testing", logger, writer, wandb_run):
        for path, module in filtered_modules:
            try:
                # Skip excluded paths
                if path in exclude_paths:
                    continue

                # Recreate flesh pipeline for each test
                pipeline = create_pipeline(model_name, device, gaudi_config)

                # Get module types
                module_type = get_submodule_type(pipeline, path)
                orig_type = get_submodule_orig_type(pipeline, path)

                # Reset pipeline state
                for mod_path, state in original_states.items():
                    mod = pipeline
                    for part in mod_path.split("."):
                        mod = getattr(mod, part)
                    mod.load_state_dict(state["state"])

                # Test module
                if mode == "compile_except":
                    # Apply compilation to all modules except this one
                    apply_compile_except(pipeline, path)
                else:  # single mode
                    # Apply compilation only to this module
                    apply_compile_to_path(pipeline, path)

                # Run test
                with torch.no_grad():
                    # Generate test image
                    output = pipeline(
                        prompt = "A picture of a dog in a bucket",
                        num_inference_steps=5
                    )

                    # Convert output to tensor if needed
                    if hasattr(output, 'images'):
                        test_image = output.images[0]
                        if not isinstance(test_image, torch.Tensor):
                            test_image = torch.from_numpy(np.array(test_image)).permute(2, 0, 1).float() / 255.0
                    else:
                        test_image = output[0]
                        if not isinstance(test_image, torch.Tensor):
                            test_image = torch.from_numpy(np.array(test_image)).permute(2, 0, 1).float() / 255.0

                    # Check if image is blank
                    is_blank_image = is_blank(test_image, std_thredhold=0.05)

                    # Save test image with status in filename
                    status = "BLANK" if is_blank_image else "OK"
                    image_path = os.path.join(images_dir, f"{path.replace('.', '_')}_{status}.png")
                    save_image_tensor(test_image, image_path)

                    if is_blank_image:
                        raise RuntimeError("Generated image is blank")

                # Log success
                result = {
                    "path": path,
                    "type": module_type,
                    "original_type": orig_type,
                    "status": "passed",
                    "error": ""
                }

            except Exception as e:
                # Log failure
                result = {
                    "path": path,
                    "type": module_type,
                    "original_type": orig_type,
                    "status": "failed",
                    "error": str(e)
                }
                bad_paths.append(path)

            results.append(result)

            # Clean up pipeline
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results, bad_paths


def main():
    """Main entry point for the diagnostic tool."""
    try:
        # Parse arguments
        args = parse_arguments()
        validate_args(args)

        # Set up logging
        logger = setup_logging(args.log_dir)
        writer = setup_tensorboard(args.tensorboard_dir) if args.use_tensorboard else None
        wandb_run = setup_wandb(args.wandb_project, get_config_from_args(args)) if args.use_wandb else None

        try:
            # List submodules
            with Timer("Submodule listing", logger, writer, wandb_run):
                submodules = list_pipeline_submodules(
                    args.model_name,
                    args.gaudi_config,
                    args.device,
                    args.output_dir
                )

            # Run diagnostic tests
            results, bad_paths = run_diagnostic(
                model_name=args.model_name,
                device=args.device,
                mode=args.mode,
                filter_type=args.filter_type,
                output_dir=args.output_dir,
                exclude_path=args.exclude_path,
                bad_paths_file=args.bad_paths_file,
                gaudi_config=args.gaudi_config,
                logger=logger,
                writer=writer,
                wandb_run=wandb_run
            )

            # Save results
            save_results(results, bad_paths, args.output_dir)

            # Print summary
            print_test_summary(results, bad_paths)

        finally:
            # Clean up logging
            cleanup_logging(logger, writer, wandb_run)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()