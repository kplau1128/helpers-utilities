"""Utility functions for working with diffusion pipelines."""

import os
import torch
import warnings
import json
from typing import Any, Optional, ContextManager
from contextlib import contextmanager
from diffusers import StableDiffusionXLPipeline
from optimum.habana.diffusers import (
    GaudiStableDiffusionXLPipeline,
    GaudiEulerDiscreteScheduler,
)


def create_pipeline(model_name, device, gaudi_config=None):
    """Create and configure a diffusion pipeline.

    This function initializes a StableDiffusionXLPipeline with the specified model
    and device, handling both HPU and CPU/GPU devices. It provides comprehensive
    error handling and configuration.

    Args:
        model_name (str): The name or path of the model to load.
        device (str): The device to run the pipeline on ("hpu", "cuda", or "cpu").
        gaudi_config (str, optional): The Gaudi configuration to use for HPU devices.

    Returns:
        StableDiffusionXLPipeline: The configured pipeline.

    Raises:
        ValueError: If the model_name or device is invalid.
        RuntimeError: If there are issues initializing the pipeline.
    """
    try:
        # Validate inputs
        if not model_name:
            raise ValueError("Model name cannot be empty")
        if device not in ["hpu", "cuda", "cpu"]:
            raise ValueError(f"Invalid device: {device}")

        # Set default Gaudi config for HPU if not provided
        if device == "hpu" and not gaudi_config:
            warnings.warn("No Gaudi config provided for HPU device, using default: Habana/stable-diffusion")
            gaudi_config = "Habana/stable-diffusion"

        # Initialize pipeline based on device
        try:
            if device == "hpu":
                scheduler = GaudiEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
                pipeline = GaudiStableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    scheduler=scheduler,
                    use_habana=True,
                    use_hpu_graphs=False,
                    gaudi_config=gaudi_config,
                    torch_dtype=torch.bfloat16
                )
            else:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16
                )
        except ImportError as e:
            raise RuntimeError(f"Failed to import required modules: {str(e)}")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to initialize pipeline: {str(e)}")

        # Configure pipeline
        try:
            # Disable progress bar
            pipeline.set_progress_bar_config(disable=True)

            # Move pipeline to device
            pipeline = pipeline.to(device)

            # Verify pipeline components
            if not hasattr(pipeline, 'vae') or not hasattr(pipeline, 'unet'):
                raise RuntimeError("Pipeline is missing required components")

            return pipeline

        except Exception as e:
            raise RuntimeError(f"Failed to configure pipeline: {str(e)}")

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Unexpected error creating pipeline: {str(e)}")


@contextmanager
def pipeline_context(model_name: str, device: str, gaudi_config: Optional[str] = None) -> ContextManager[Any]:
    """Context manager for pipeline creation and cleanup.

    Args:
        model_name (str): Name of the model to create.
        device (str): Device to run the model on.
        gaudi_config (Optional[str]): Path to Gaudi configuration file.

    Yields:
        Any: The created pipeline.
    """
    pipeline = create_pipeline(model_name, device, gaudi_config)
    try:
        yield pipeline
    finally:
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_results(results, bad_paths, output_dir):
    """Save diagnostic results to files.

    This function saves the diagnostic results to CSV and text files in the
    specified output directory. It handles directory creation and provides
    comprehensive error handling.

    Args:
        results (list): List of dictionaries containing diagnostic results.
        bad_paths (list): List of paths that failed the diagnostic test.
        output_dir (str): Directory to save the results in.

    Raises:
        ValueError: If the input data is invalid.
        IOError: If there are issues creating directories or saving files.
    """
    try:
        # Validate inputs
        if not isinstance(results, list):
            raise ValueError("Results must be a list")
        if not isinstance(bad_paths, list):
            raise ValueError("Bad paths must be a list")

        # Create output directory
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            raise IOError(f"Failed to create output directory: {str(e)}")

        # Save results to CSV
        try:
            csv_path = os.path.join(output_dir, "results.csv")
            with open(csv_path, "w") as f:
                # Write header
                f.write("path,type,original_type,status,error\n")

                # Write results
                for result in results:
                    if not isinstance(result, dict):
                        raise ValueError("Each result must be a dictionary")

                    # Get values with defaults
                    path = result.get("path", "")
                    type_str = str(result.get("type", ""))
                    orig_type = str(result.get("original_type", ""))
                    status = result.get("status", "unknown")
                    error = str(result.get("error", "")).replace(",", ";")

                    # Write row
                    f.write(f"{path},{type_str},{orig_type},{status},{error}\n")

        except Exception as e:
            raise IOError(f"Failed to save results to CSV: {str(e)}")

        # Save results to JSON
        try:
            json_path = os.path.join(output_dir, "results.json")
            with open(json_path, "w") as f:
                json.dump({
                    "results": results,
                    "bad_paths": bad_paths,
                    "summary": {
                        "total": len(results),
                        "failed": len(bad_paths),
                        "passed": len(results) - len(bad_paths)
                    }
                }, f, indent=4)

        except Exception as e:
            raise IOError(f"Failed to save results to JSON: {str(e)}")

        # Save bad paths to text file
        try:
            bad_paths_file = os.path.join(output_dir, "bad_paths.txt")
            with open(bad_paths_file, "w") as f:
                for path in bad_paths:
                    if not isinstance(path, str):
                        raise ValueError("Each path must be a string")
                    f.write(f"{path}\n")

        except Exception as e:
            raise IOError(f"Failed to save bad paths: {str(e)}")

    except Exception as e:
        if isinstance(e, (ValueError, IOError)):
            raise
        raise IOError(f"Unexpected error saving results: {str(e)}")


def print_test_summary(results, bad_paths):
    """Print a summary of the diagnostic test results.

    This function prints a formatted summary of the test results, including
    statistics and details about failed paths. It provides comprehensive error
    handling.

    Args:
        results (list): List of dictionaries containing diagnostic results.
        bad_paths (list): List of paths that failed the diagnostic test.

    Raises:
        ValueError: If the input data is invalid.
    """
    try:
        # Validate inputs
        if not isinstance(results, list):
            raise ValueError("Results must be a list")
        if not isinstance(bad_paths, list):
            raise ValueError("Bad paths must be a list")

        # Calculate statistics
        try:
            total = len(results)
            passed = sum(1 for r in results if r.get("status") == "passed")
            failed = sum(1 for r in results if r.get("status") == "failed")
            unknown = total - passed - failed

            # Print summary
            print("\nTest Summary:")
            print(f"Total modules tested: {total}")
            print(f"Passed: {passed}")
            print(f"Failed: {failed}")
            print(f"Unknown: {unknown}")

        except Exception as e:
            print(f"Warning: Failed to calculate statistics: {str(e)}")

        # Print failed paths
        try:
            if bad_paths:
                print("\nFailed Paths:")
                for path in bad_paths:
                    if not isinstance(path, str):
                        raise ValueError("Each path must be a string")
                    print(f"- {path}")

        except Exception as e:
            print(f"Warning: Failed to print failed paths: {str(e)}")

        # Print error details
        try:
            errors = [r for r in results if r.get("status") == "failed" and r.get("error")]
            if errors:
                print("\nError Details:")
                for error in errors:
                    path = error.get("path", "unknown")
                    error_msg = error.get("error", "unknown error")
                    print(f"- {path}: {error_msg}")

        except Exception as e:
            print(f"Warning: Failed to print error details: {str(e)}")

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        print(f"Warning: Failed to print test summary: {str(e)}")


def list_pipeline_submodules(model_name, gaudi_config, device, output_dir):
    """List all submodules of a pipeline and save them to a file.

    This function creates a pipeline, lists its submodules, and saves the list
    to both text and JSON files. It provides comprehensive error handling.

    Args:
        model_name (str): The name or path of the model to load.
        gaudi_config (str, optional): The Gaudi configuration to use for HPU devices.
        device (str): The device to run the pipeline on.
        output_dir (str): Directory to save the submodule list in.

    Returns:
        list: List of submodule paths and types.

    Raises:
        RuntimeError: If there are issues with the pipeline or file operations.
    """
    try:
        # Inner function to format module line
        def format_module_line(current_module, prefix, is_last, vertical_lines, extra_info=""):
            """Format a single module line with indentation and markers."""
            name = prefix.split('.')[-1] if prefix else type(current_module).__name__
            type_str = type(current_module).__name__
            marker = '└── ' if is_last else '├── '
            indent = ''.join('│   ' if show_line else '    ' for show_line in vertical_lines)
            return f"{indent}{marker}{name} ({type_str}){extra_info}"

        # Inner function to format the submodule tree
        def format_submodule_tree(module):

            def get_children(current_module):
                """Get all child modules of the current module."""
                children = []
                try:
                    # First try named_children() for direct children
                    children.extend(current_module.named_children())
                except Exception:
                    pass

                # Get all attributes that are modules
                for attr_name in dir(current_module):
                    if attr_name.startswith('_'):
                        continue
                    try:
                        attr = getattr(current_module, attr_name)
                        # Exclude self-reference
                        if attr == current_module:
                            continue
                        if hasattr(attr, 'named_modules') and not isinstance(attr, type):
                            # Only add if not already in children list
                            if not any(c[0] == attr_name for c in children):
                                children.append((attr_name, attr))
                    except Exception:
                        continue
                return children

            # Generate text tree
            def build_text_tree(root_module):
                """Build the text representation of the module tree using a stack."""
                output = []
                # Stack of (module, prefix, is_last, depth, vertical_lines) tuples
                # vertical_lines is a list of booleans indicating whether to show vertical line at each level
                stack = [(root_module, '', True, 0, [])]
                # Track visited modules and their paths
                visited = {}

                while stack:
                    current_module, prefix, is_last, depth, vertical_lines = stack.pop()

                    # Check for cyclic references and reused modules
                    if id(current_module) in visited:
                        if prefix in visited[id(current_module)]:
                            output.append(format_module_line(current_module, prefix, is_last, vertical_lines, " [Cyclic Reference Deteected]"))
                            continue
                        else:
                            # mark as reused module if visited at a different path
                            visited[id(current_module)].append(prefix)
                            output.append(format_module_line(current_module, prefix, is_last, vertical_lines, " [Reused Module]"))
                            continue
                    else:
                        # First time visiting this module
                        visited[id(current_module)] = [prefix]
                        output.append(format_module_line(current_module, prefix, is_last, vertical_lines))

                    # Get children
                    children = get_children(current_module)

                    # Add children to stack in reverse order (so they're processed in correct order)
                    for i, (name, child) in enumerate(reversed(children)):
                        is_last_child = i == 0  # First in reversed list is last in original
                        full_name = f"{prefix}.{name}" if prefix else name
                        # For children, we need to show vertical line if current node is not last
                        child_vertical_lines = vertical_lines + [not is_last]
                        stack.append((child, full_name, is_last_child, depth + 1, child_vertical_lines))
                return output

            # Generate JSON tree
            json_visited = {}

            def build_json_tree(current_module, prefix, is_last, depth, vertical_lines):
                """Build the JSON representation of the module tree."""
                module_dict = {
                    "name": prefix.split('.')[-1] if prefix else type(current_module).__name__,
                    "type": type(current_module).__name__,
                    "path": prefix,
                    "is_last": is_last,
                    "depth": depth,
                    "children": []
                }

                # Detect cyclic references and reuse modules
                if id(current_module) in json_visited:
                    if prefix in json_visited[id(current_module)]:
                        module_dict["status"] = "cyclic_reference"
                        return module_dict
                    else:
                        # mark as reused module if visited at a different path
                        json_visited[id(current_module)].append(prefix)
                        module_dict["status"] = "reused"
                        return module_dict
                else:
                    # First time visiting this module
                    json_visited[id(current_module)] = [prefix]
                    module_dict["status"] = "normal"

                # Get and process children
                children = get_children(current_module)
                for i, (name, child) in enumerate(children):
                    is_last_child = i == len(children) - 1
                    full_name = f"{prefix}.{name}" if prefix else name
                    # For children, we need to show vertical line if current node is not last
                    child_vertical_lines = vertical_lines + [not is_last]
                    child_dict = build_json_tree(child, full_name, is_last_child, depth + 1, child_vertical_lines)
                    module_dict["children"].append(child_dict)

                return module_dict

            # Build both trees
            text_output = build_text_tree(module)
            json_output = build_json_tree(module, '', True, 0, [])

            return text_output, json_output

        def count_modules(module_dict):
            """Count modules in the tree and collect statistics."""
            stats = {
                "total": 1,  # Count current module
                "paths": {module_dict["path"]},
                "reused": 1 if module_dict.get("status") == "reused" else 0,
                "cyclic": 1 if module_dict.get("status") == "cyclic_reference" else 0
            }

            # Recursively count children
            for child in module_dict["children"]:
                child_stats = count_modules(child)
                stats["total"] += child_stats["total"]
                stats["paths"].update(child_stats["paths"])
                stats["reused"] += child_stats["reused"]
                stats["cyclic"] += child_stats["cyclic"]

            return stats

        # Generate tree structure using pipeline context manager
        with pipeline_context(model_name, device, gaudi_config) as pipeline:
            tree_output, json_output = format_submodule_tree(pipeline)

            # Calculate statistics from the JSON tree
            stats = count_modules(json_output)

            # Save to files
            try:
                os.makedirs(output_dir, exist_ok=True)

                # Save text format
                output_file = os.path.join(output_dir, "submodules.txt")
                with open(output_file, "w") as f:
                    f.write("Pipeline Submodules:\n")
                    f.write("====================\n")
                    f.writelines("\n".join(tree_output))

                # Save JSON format
                json_file = os.path.join(output_dir, "submodules.json")
                with open(json_file, "w") as f:
                    json.dump({
                        "model_name": model_name,
                        "device": device,
                        "gaudi_config": gaudi_config,
                        "module_tree": json_output,
                        "statistics": {
                            "total_modules": stats["total"],
                            "unique_modules": len(stats["paths"]),
                            "reused_modules": stats["reused"],
                            "cyclic_references": stats["cyclic"]
                        }
                    }, f, indent=4)

            except Exception as e:
                raise RuntimeError(f"Failed to save submodule list: {str(e)}")

            return tree_output

    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"Unexpected error listing submodules: {str(e)}")
