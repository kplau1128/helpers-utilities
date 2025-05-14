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
import logging
from torch.utils.tensorboard import SummaryWriter
import wandb

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
    convert_to_tensor,

    # Pipeline utilities
    pipeline_context,
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
    test_paths: Optional[str] = None,
    gaudi_config: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    writer: Optional[SummaryWriter] = None,
    wandb_run: Optional[wandb.run] = None,
    root_modules: Optional[List[str]] = None,
    test_prompt: str = "A picture of a dog in a bucket",
    num_inference_steps: int = 50,
    std_threshold: float = 0.05
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Run diagnostic tests on a diffusion model.

    Args:
        model_name (str): Name of the model to test.
        device (str): Device to run the model on.
        mode (str): Test mode ("single" or "compile_except").
        filter_type (str): Type of modules to test ("all", "leaf", or "non-leaf").
        output_dir (str): Directory to save test results.
        exclude_path (Optional[str]): Path(s) to exclude from testing. Can be a comma-separated list of paths or a file containing paths to exclude.
        test_paths (Optional[str]): Specific path(s) to test. Can be a comma-separated list of paths or a file containing paths to test.
        gaudi_config (Optional[str]): Path to Gaudi configuration file.
        logger (Optional[logging.Logger]): Logger instance.
        writer (Optional[SummaryWriter]): TensorBoard writer.
        wandb_run (Optional[wandb.run]): Weights & Biases run.
        root_modules (Optional[List[str]]): List of paths to modules to use as roots for diagnostics. If None, uses the entire pipeline.
        test_prompt (str): The prompt to use for testing the model.
        num_inference_steps (int): Number of inference steps to run for each test.
        std_threshold (float): Threshold for determining if an image is blank.

    Returns:
        tuple[List[Dict[str, Any]], List[str]]: Test results and bad paths.
    """
    # Initialize results and bad paths
    results = []
    bad_paths = []

    # Get paths to test
    paths_to_test = []
    if test_paths:
        # Check if test_paths is a file
        if os.path.exists(test_paths) and os.path.isfile(test_paths):
            with open(test_paths, 'r') as f:
                paths_to_test.extend([line.strip() for line in f if line.strip()])
        else:
            # Assume it's a comma-separated list of paths
            paths_to_test.extend([path.strip() for path in test_paths.split(',') if path.strip()])

    # Get excluded paths
    exclude_paths = []
    if exclude_path:
        # Check if exclude_path is a file
        if os.path.exists(exclude_path) and os.path.isfile(exclude_path):
            with open(exclude_path, 'r') as f:
                exclude_paths.extend([line.strip() for line in f if line.strip()])
        else:
            # Assume it's a comma-separated list of paths
            exclude_paths.extend([path.strip() for path in exclude_path.split(',') if path.strip()])
    else:
        # Try to load from bad_paths.txt if it exists
        bad_paths_file = os.path.join(output_dir, "bad_paths.txt")
        if os.path.exists(bad_paths_file):
            with open(bad_paths_file, 'r') as f:
                exclude_paths.extend([line.strip() for line in f if line.strip()])

    # Create initial pipeline and get modules
    with pipeline_context(model_name, device, gaudi_config) as init_pipeline:
        # If root_modules is specified, get those modules as roots
        if root_modules:
            all_modules = []
            for root_module in root_modules:
                root = init_pipeline
                for part in root_module.split('.'):
                    root = getattr(root, part)
                root_submodules = list(get_all_submodules(root))
                # Adjust paths to be relative to the root module
                root_submodules = [(f"{root_module}.{path}" if path else root_module, module) for path, module in root_submodules]
                all_modules.extend(root_submodules)
        else:
            all_modules = list(get_all_submodules(init_pipeline))

        # Store original module states for resetting
        original_states = {}
        for path, module in all_modules:
            original_states[path] = {
                "state": copy.deepcopy(module.state_dict()),
                "type": type(module)
            }

    # Filter modules based on test_paths if provided
    if paths_to_test:
        filtered_modules = [(path, module) for path, module in all_modules if path in paths_to_test]
    else:
        filtered_modules = filter_submodules(all_modules, filter_type)

    # Create images subdirectory
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Run tests
    with Timer("Testing", logger, writer, wandb_run):
        if mode == "compile_except":
            # Create fresh pipeline for compile_except mode
            with pipeline_context(model_name, device, gaudi_config) as pipeline:
                try:
                    # Apply compilation to all modules except excluded paths
                    pipeline = apply_compile_except(pipeline, exclude_paths)
                    
                    # Run test with compiled pipeline
                    with torch.no_grad():
                        # Generate test image
                        output = pipeline(
                            prompt=test_prompt,
                            num_inference_steps=num_inference_steps
                        )

                        # Convert output to tensor
                        if hasattr(output, 'images'):
                            test_image = convert_to_tensor(output.images[0])
                        else:
                            test_image = convert_to_tensor(output[0])

                        # Check if image is blank
                        is_blank_image = is_blank(test_image, std_threshold=std_threshold)

                        # Save test image with status in filename
                        status = "BLANK" if is_blank_image else "OK"
                        paths_str = "_".join(p.replace('.', '_') for p in exclude_paths) if exclude_paths else "all"
                        image_path = os.path.join(images_dir, f"except_{paths_str}_{status}.png")
                        save_image_tensor(test_image, image_path)

                        if is_blank_image:
                            raise RuntimeError("Generated image is blank")

                    # Log success
                    result = {
                        "mode": "compile_except",
                        "paths": exclude_paths,
                        "status": "passed",
                        "error": ""
                    }

                except Exception as e:
                    # Log failure with detailed error information
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    if logger:
                        logger.error(f"Test failed for compile_except mode: {error_msg}")
                    
                    result = {
                        "mode": "compile_except",
                        "paths": exclude_paths,
                        "status": "failed",
                        "error": error_msg
                    }
                    # Add all excluded paths to bad_paths
                    bad_paths.extend(exclude_paths)

                results.append(result)

        else:  # single mode
            for path, module in filtered_modules:
                try:
                    # Skip excluded paths
                    if path in exclude_paths:
                        continue

                    # Create fresh pipeline for each test
                    with pipeline_context(model_name, device, gaudi_config) as pipeline:
                        # Get module types
                        module_type = get_submodule_type(pipeline, path)
                        orig_type = get_submodule_orig_type(pipeline, path)

                        # Reset pipeline state
                        for mod_path, state in original_states.items():
                            mod = pipeline
                            for part in mod_path.split("."):
                                mod = getattr(mod, part)
                            mod.load_state_dict(state["state"])

                        # Apply compilation only to this module
                        apply_compile_to_path(pipeline, path)

                        # Run test
                        with torch.no_grad():
                            # Generate test image
                            output = pipeline(
                                prompt=test_prompt,
                                num_inference_steps=num_inference_steps
                            )

                            # Convert output to tensor
                            if hasattr(output, 'images'):
                                test_image = convert_to_tensor(output.images[0])
                            else:
                                test_image = convert_to_tensor(output[0])

                            # Check if image is blank
                            is_blank_image = is_blank(test_image, std_threshold=std_threshold)

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
                    # Log failure with detailed error information
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    if logger:
                        logger.error(f"Test failed for path {path}: {error_msg}")
                    
                    result = {
                        "path": path,
                        "type": module_type,
                        "original_type": orig_type,
                        "status": "failed",
                        "error": error_msg
                    }
                    bad_paths.append(path)

                results.append(result)

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
                test_paths=args.test_paths,
                gaudi_config=args.gaudi_config,
                logger=logger,
                writer=writer,
                wandb_run=wandb_run,
                root_modules=args.root_modules.split(',') if args.root_modules else None,
                test_prompt=args.test_prompt if hasattr(args, 'test_prompt') else "A picture of a dog in a bucket",
                num_inference_steps=args.num_inference_steps if hasattr(args, 'num_inference_steps') else 50,
                std_threshold=args.std_threshold if hasattr(args, 'std_threshold') else 0.05
            )

            # Save results
            save_results(results, bad_paths, args.output_dir)

            # Print summary
            print_test_summary(results, bad_paths)

            # Log metrics if enabled
            if writer or wandb_run:
                log_metrics(results, writer, wandb_run)

        except Exception as e:
            error_msg = f"Error during diagnostic testing: {type(e).__name__}: {str(e)}"
            if logger:
                logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        finally:
            # Cleanup
            cleanup_logging(logger, writer, wandb_run)

    except Exception as e:
        print(f"Fatal error: {type(e).__name__}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
