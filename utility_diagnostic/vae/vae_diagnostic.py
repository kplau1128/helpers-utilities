import torch
import torch.nn as nn
import argparse
import json
import csv
import os
import copy
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

# Check if weights and Biases (wandb) is available for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False

from diffusers import StableDiffusionXLPipeline
from optimum.habana.diffusers import (
    GaudiStableDiffusionXLPipeline,
    GaudiEulerDiscreteScheduler,
)


class CompiledWrapper(nn.Module):
    """A wrapper class to compile a PyTorch module using the HPU backend.

    This class encapsulates a PyTorch module, compiles it with the specified backend
    (hpu_backend), and presserves the original module type for diagnostic purposes.

    Args:
        module (nn.Module): The PyTorch module to be compiled.
    """
    def __init__(self, module):
        super().__init__()
        # Compile the module with the HPU backend for optimized execution
        self.compiled = torch.compile(module, backend="hpu_backend")
        # Sore the original module type for reference
        self.original_type = type(module)

    def forward(self, *args, **kwargs):
        """Forward pass through the compiled module.

        Args:
            *args: Variable length argument list passed to the compiled module.
            **kwargs: Arbitary keyword arguments passed to the compiled module.

        Returns:
            The output of the compiled module's forward pass.
        """
        return self.compiled(*args, **kwargs)


def get_all_submodules(module, prefix=""):
    """Recursively retrieve all sumodules of a PyTorch module with their hierarchical paths.

    This generator function traverses the module's hierarchy and yields tuples containing
    the path and submodule for each submodule encountered.

    Args:
        module (nn.Module): The PyTorch module to traverse.
        prefix (str, optional): The prefix for the current module path. Defaults to "".

    Yields:
        tuple: A tuple containing the path (str) and submodule (nn.Module).
    """
    for name, sub in module.named_children():
        # Construct the full path for the current submodule
        path = f"{prefix}.{name}" if prefix else name
        yield path, sub
        # Recursively yield submodules of the current submodule
        yield from get_all_submodules(sub, path)


def filter_submodules(modules, filter_type):
    """Filter a list of submodules based on the specified filter type.

    Args:
        modules (list): List of tuples containing (path, submodule).
        filter_type (str): Type of filter to apply ("all", "leaf", or "non-leaf").

    Returns:
        list: Filtered list of (path, submodule) tuple.

    Raises:
        ValueError: If filter_type is not one of "all", "leaf", or "non-left".
    """
    if filter_type == "all":
        # Return all modules without filtering
        return modules
    elif filter_type == "leaf":
        # Return only leaf modules (those with no children)
        return [(p, m) for p, m in modules if not any(m.children())]
    elif filter_type == "non-leaf":
        # Return only nonh-leaf modules (those with children)
        return [(p, m) for p, m in modules if any(m.children())]
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def is_wrappable_module(module):
    """Check if a module can be wrapped for compilation.

    Certain module types (e.g., ModuleList, ModuleDict) are not suitable for wrapping
    due to their structure or behavior.

    Args:
        module (nn.Module): The PyTorch module to check.

    Returns:
        bool: True if the module can be wrapped, False otherwise.
    """
    # Define non-wrappable module types
    non_wrappable_types = (nn.ModuleList, nn.ModuleDict, nn.Sequential)
    return isinstance(module, nn.Module) and not isinstance(module, non_wrappable_types)


def apply_compile_to_path(module, target_path, prefix=""):
    """Apply compilation to a specific submodule at the given path.

    This function recursively traverses the module hierarchy and wraps the target
    submodule with CompiledWrapper if it is warppable.

    Args:
        module (nn.Module): The root PyTorch module to traverse.
        target_path (str): The path to the target submodule.
        prefix (str, optional): The current path prefix. Defaults to "".
    """
    for name, sub in module.named_children():
        # Construct the full path for the current submodule
        path = f"{prefix}.{name}" if prefix else name
        if path == target_path and callable(sub) and is_wrappable_module(sub):
            # Replace the target submodule with its compiled wrapper
            setattr(module, name, CompiledWrapper(sub))
        else:
            # Recursively traverse children
            apply_compile_to_path(sub, target_path, path)


def apply_compile_except(module, skip_path, prefix=""):
    """Apply compilation to all submodules except the one at the specified path.

    This function recursively traverses the module hierarchy and wraps all wrappable
    submodules except the one at skip_path with a CompiledWrapper.

    Args:
        module (nn.Module): The root PyTorch module to traverse.
        skip_path (str): The path of the submodule to skip.
        prefix (str, optional): The current path prefix. Defaults to "".
    """
    for name, sub in module.named_children():
        # Construct the full path for the current submodule
        path = f"{prefix}.{name}" if prefix else name
        if path != skip_path and callable(sub) and is_wrappable_module(sub):
            # Replace the submodule with its Compiled wrapper
            setattr(module, name, CompiledWrapper(sub))
        else:
            # Recursively traverse children
            apply_compile_except(sub, skip_path, path)


def get_submodule_type(model, path):
    """Get the type of a submodule at specified path.

    Args:
        model (nn.Module): The root PyTorch module.
        path (str): The path to the submodule.

    Returns:
        type: The type of the submodule.
    """
    submod = model
    # Traverse the path to reach the target submodule
    for part in path.split('.'):
        submod = getattr(submod, part)
    return type(submod)


def get_submodule_orig_type(model, path):
    """Get the original type of a submodule at specified path.

    Args:
        model (nn.Module): The root PyTorch module.
        path (str): The path to the submodule.

    Returns:
        type: The original type of the submodule.
    """
    submod = model
    # Traverse the path to reach the target submodue
    for part in path.split('.'):
        submod = getattr(submod, part)
    # Return the original type if wrapped, otherwise the current type
    return getattr(submod, 'original_type', type(submod))


def is_blank(tensor,
             std_thredhold=1e-4,
             constant_value=None,
             outlier_tolerance=0.0):
    """Check if an image tensor is blank (all pixels are almost the same, or equal to a specific value).

    This function supports both torch.Tensor and PIL.Image inputs, and handles
    different tensor dimensions (batch and single images). It validates the input
    and provides comprehensive error handling.

    Args:
        tensor (Union[torch.Tensor, PIL.Image]): The image tensor or PIL Image to check.
        std_threshold (float): Standard deviation threshold for blankness.
        constant_value 9float orNone): If set, check if all pixels are (almost) this value.
        outlier_tolerance (float): Fraction of pixels allowed to differ to from the constant value (0.0 = strict)

    Returns:
        bool: True if the image is blank, False otherwise.

    Raises:
        ValueError: If the input is invalid or has unexpected properties.
    """
    try:
        # Convert PIL Image to tensor if necessary
        if isinstance(tensor, Image.Image):
            tensor = ToTensor()(tensor)

        # Validate tensor type
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor or PIL.Image")

        # Check if tensor is empty
        if tensor.numel() == 0:
            raise ValueError("Input tensor is empty")

        # Handle batch dimension
        if tensor.dim() == 4:  # Batch of images
            tensor = tensor[0]  # Take first image
        elif tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor (C,H,W) or 4D tensor (B,C,H,W), got {tensor.dim()}D")

        # Validate number of channels
        if tensor.size(0) not in [1, 3]:
            raise ValueError(f"Expected 1 or 3 channels, got {tensor.size(0)}")

        # Option 1: Check for standard deviation (general blankness)
        if constant_value is None:
            return tensor.std().item() < std_thredhold

        # Optiion 2: Check for a specific constant value, allowing some tolerance for outliers
        diff = torch.abs(tensor - constant_value)
        num_outliers = torch.sum(diff > std_thredhold).item()
        total = tensor.numel()
        return (num_outliers / total) <= outlier_tolerance

    except Exception as e:
        raise ValueError(f"Error checking if image is blank: {str(e)}")


def save_image_tensor(image_tensor, path):
    """Save an image tensor to a file.
    
    Args:
        image_tensor (PIL.Image or torch.Tensor): The image to save.
        path (str): The file path to save the image to.
    """
    # Save the image to the specified path
    image_tensor.save(path)


def list_submodules(model, prefix='', depth=0):
    """List all submodules of a model hierarchically.
    This function generates a list of strings describing the submodule hierarchy,
    including their paths and types. It handles cases where some submodules might
    not have named_children() function.

    Args:
        model (nn.Module): The PyTorch module to traverse.
        prefix (str, optional): The current path prefix. Defaults to "".
        depth (int, optional): The current depth in the hierarchy. Defaults to 0.

    Returns:
        list: A list of strings describing the submodules.
    """
    lines = []
    # Stack of (module, prefix, is_last, depth, vertical_lines) tuples
    # vertical_lines is a list of booleans indicating whether to show vertical line at each level
    stack = [(model, '', True, 0, [])]
    # Track visited modules and their paths
    visited = {}
    
    def format_module_line(current_module, prefix, is_last, vertical_lines, extra_info=""):
        """Format a single module line with indentation and markers."""
        name = prefix.split('.')[-1] if prefix else type(current_module).__name__
        type_str = type(current_module).__name__
        marker = '└── ' if is_last else '├── '
        indent = ''.join('│   ' if show_line else '    ' for show_line in vertical_lines)
        return f"{indent}{marker}{name} ({type_str}){extra_info}"
    
    while stack:
        current_module, prefix, is_last, depth, vertical_lines = stack.pop()
        
        # Detect cyclic references
        if id(current_module) in visited:
            if prefix in visited[id(current_module)]:
                # mark as cyclic reference if the same path is visited again
                lines.append(format_module_line(current_module, prefix, is_last, vertical_lines, " [Cyclic Reference Deteected]"))
                continue
            else:
                # mark as reused module if visited at a different path
                lines.append(format_module_line(current_module, prefix, is_last, vertical_lines, " [Reused Module]"))
                continue
        else:
            # First time visiting this module
            visited[id(current_module)] = [prefix]

        children = []
        try:
            children = list(current_module.named_children())
        except Exception:
            children = []

        # Sort children for consistent output
        # children.sort(key=lambda x: x[0])
        
        # Process current module if it's not the root
        lines.append(format_module_line(current_module, prefix, is_last, vertical_lines))

        # Add children to stack in reverse order
        for i, (name, child) in enumerate(reversed(children)):
            is_last_child = i == 0  # First in reversed list is last in original
            full_name = f"{prefix}.{name}" if prefix else name
            # For children, we need to show vertical line if current node is not last
            child_vertical_lines = vertical_lines + [not is_last]
            stack.append((child, full_name, is_last_child, depth + 1, child_vertical_lines))
    
    return lines


# --- Create Pipeline ---
def create_pipeline(model_name, gaudi_config=None, device="cpu"):
    """Create a diffusion pipeline based on the specified device.

    This function initializes either a gaudi-optimized pipeline for HPU devices
    or a standard Stable Diffusion pipeline for CPU/GPU devices.

    Args:
        model_name (str): The name of the pretrained model to load.
        gaudi_config(str, optional): Configuration for Gaudi pipeline. Defaults to None.
        device (str, optional): The device to run the pipeline on ("hpu or "cpu"). Defaults to "cpu".

    Returns:
        Pipeline: The initialized diffusion pipeline.
    """
    if device == "hpu":
        # Initialize Gaudi-optimized pipeline for HPU
        print("[LOAD] GaudiStableDiffusionXLPipeline...")
        scheduler = GaudiEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipe = GaudiStableDiffusionXLPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=False,
            gaudi_config=gaudi_config,
            torch_dtype=torch.bfloat16,
        )
    else:
        # Initialize standard pipeline for CPU/GPU
        print("[LOAD] StableDiffusionXLPipeline...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
        )

    # Disable progress bar for cleaner output
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


# ------------------ Main Diagnostic Runner ------------------
def run_diagnostic(model_name, gaudi_config, device, mode, filter_type, exclude_path,
                   save_images, output_dir, tb_writer, wandb_run, submodules=None):
    """Run diagnostic tests on the VAE decoder submodules of a diffusion pipeline.
    
    The function tests the effect of compiling individual submodules or all submodules
    except a specified path, generating images and checking for blank outputs or errors.
    
    Args:
        model_name (str): The name of the pretrained model.
        gaudi_config (str): Configuration for Gaudi pipeline.
        device (str): The device to run the pipeline on ("hpu" or "cpu").
        mode (str): Compilation mode {"single", or "compile_except").
        filter_type (str): Type of submodules to test ("all", "leaf", or "non-leaf").
        exclude_path (str or list): Path(s) to exclude in compile_except mode.
        save_images (bool): Whether to save generated images.
        output_dir (str): Directory to save diagnostic results.
        tb_writer (SummaryWriter): TensorBoard writer for logging.
        wandb_run (wandb.Run): Weights and Biases run for logging.
        submodules (list, optional): Pre-filtered list of (path, submodule) tuples to test.
        
    Returns:
        tuple: A tuple containing:
            - results (list): List of diagnostic results for each test.
            - bad_paths (list): List of paths that produced blank images or errors.
    """
    # Initialize the diffusion pipeline
    pipe = create_pipeline(model_name, gaudi_config, device)
    # Store the original VAE decoder for resetting between tests
    vae_decoder_orig = pipe.vae.decoder
    
    # Get submodules to test
    if submodules is None:
        # Get all submodules of VAE decoder if no pre-filtered list provided
        all_submodules = list(get_all_submodules(pipe.vae.decoder))
        # Filter submodules based on the specified type
        submodules = filter_submodules(all_submodules, filter_type)
    
    if save_images:
        # Create images subdirectory
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

    results = []
    bad_paths = []
    # Define test prompt and infernecesteps
    prompt = "A picture of a dog in a bucket"
    num_inference_steps=25

    if mode == "compile_except":
        if os.path.exists(exclude_path):
            # If exclude_path is a file path, read paths from file
            with open(exclude_path, "r") as f:
                exclude_paths = [line.strip() for line in f if line.strip()]
        else:
            # Convert exclude_path to list if it's a string
            exclude_paths = [exclude_path] if isinstance(exclude_path, str) else exclude_path
        print(f"[MODE] Compiling all except: {', '.join(exclude_paths)}")
        pipe.vae.decoder = copy.deepcopy(vae_decoder_orig).to(device)
        try:
            # Apply compile_except for each path
            for path in exclude_paths:
                apply_compile_except(pipe.vae.decoder, path)
            # Generate an image using the pipeline
            image = pipe(prompt=prompt, num_inference_steps=num_inference_steps).images[0]
            img_tensor = ToTensor()(image)
            blank = is_blank(img_tensor, std_thredhold=0.05)
            result = {
                "mode": "compile_except",
                "paths": exclude_paths,
                "blank": blank,
                "mean": img_tensor.mean().item(),
                "std": img_tensor.std().item(),
                "error": ""
            }
            if blank or result["mean"] == 0:
                bad_paths.extend(exclude_paths)
            if save_images:
                status = "BLANK" if blank else "OK"
                paths_str = "_".join(p.replace('.', '_') for p in exclude_paths)
                save_image_tensor(image, os.path.join(images_dir, f"except_{paths_str}_{status}.png"))
            if tb_writer:
                # Log metrics to TensorBoard
                for path in exclude_paths:
                    tb_writer.add_scalar("mean", result["mean"], f"except_{path}")
                    tb_writer.add_scalar("std", result["std"], f"except_{path}")
            if wandb_run:
                # Log metrics to Weights and Biases
                for path in exclude_paths:
                    wandb_run.log({f"{path}/mean": result["mean"],
                                   f"{path}/std": result["std"],
                                   f"{path}/blank": blank})
        except Exception as e:
            err_msg = str(e)
            if "is not iterable" in str(e) and "CompiledWrapper" in str(e):
                # Try to get the original type for each path
                type_info = []
                for path in exclude_paths:
                    try:
                        orig_type = get_submodule_orig_type(pipe.vae.decoder, path)
                        type_info.append(f"{path} ({orig_type.__module__}.{orig_type.__name__})")
                    except:
                        type_info.append(path)
                err_msg += f" ({', '.join(type_info)})"
            result = {
                "mode": "compile_except",
                "paths": exclude_paths,
                "blank": True,
                "mean": 0,
                "std": 0,
                "error": err_msg
            }
            bad_paths.extend(exclude_paths)
        results.append(result)
    else:
        # Test compilation of individual submodules
        for path, _ in tqdm(submodules, desc="Testing submodules"):
            pipe.vae.decoder = copy.deepcopy(vae_decoder_orig).to(device)
            try:
                apply_compile_to_path(pipe.vae.decoder, path)
                # Generate an image using the pipeline
                image = pipe(prompt=prompt, num_inference_steps=num_inference_steps).images[0]
                img_tensor = ToTensor()(image)
                blank = is_blank(img_tensor, std_thredhold=0.05)
                result = {
                    "mode": "single",
                    "path": path,
                    "blank": blank,
                    "mean": img_tensor.mean().item(),
                    "std": img_tensor.std().item(),
                    "error": ""
                }
                if blank or result["mean"] == 0:
                    bad_paths.append(path)
                if save_images:
                    status = "BLANK" if blank else "OK"
                    save_image_tensor(image, os.path.join(images_dir, f"{path.replace('.', '_')}_{status}.png"))
                if tb_writer:
                    # Log metrics to TensorBoard
                    tb_writer.add_scalar("mean", result["mean"], path)
                    tb_writer.add_scalar("std", result["std"], path)
                if wandb_run:
                    # Log metrics to Weights and Biases
                    wandb_run.log({f"{path}/mean": result["mean"],
                                   f"{path}/std": result["std"],
                                   f"{path}/blank": blank})
            except Exception as e:
                err_msg = str(e)
                if "is not iterable" in str(e) and "CompiledWrapper" in str(e):
                    orig_type = get_submodule_orig_type(pipe.vae.decoder, path)
                    err_msg += f" ({orig_type.__module__}.{orig_type.__name__})"
                result = {
                    "mode": "single",
                    "path": path,
                    "blank": True,
                    "mean": 0,
                    "std": 0,
                    "error": err_msg
                }
                bad_paths.append(path)
            results.append(result)

    return results, bad_paths


def save_results(results, bad_paths, json_path, csv_path, bad_paths_path):
    """Save diagnostic results to JSON, CSV, and bad paths to a text file.

    Args:
        results (list): List of diagnostic result dictionaries.
        bad_paths (list): List of paths that produced blank image or errors.
        json_path (str): Path to save the JSON results file.
        csv_path (str): Path to save the CSV results file.
        bad_paths_path (str): Path to save the bad paths text file.
    """
    # Save results as JSON with indentation for readability
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save results as CSV with headers
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Save bad paths to a text file, one per line
    with open(bad_paths_path, "w") as f:
        for path in bad_paths:
            f.write(path + "\n")


def print_test_summary(results, bad_paths, output_dir):
    """Print a summary of the diagnostic test results.
    
    Args:
        results (list): List of diagnostic result dictionaries.
        bad_paths (list): List of paths that produced blank images or errors.
        output_dir (str): Directory where results are saved.
    """
    total_tests = len(results)
    blank_outputs = sum(1 for r in results if r["blank"])
    errors = sum(1 for r in results if r["error"])
    successful = total_tests - blank_outputs - errors

    print("\n===== Test Summary =====")
    print(f"Total tests run: {total_tests}")
    print(f"Successful tests: {successful}")
    print(f"Blank outputs: {blank_outputs}")
    print(f"Errors: {errors}")
    
    if bad_paths:
        print("\nProblematic submodules:")
        for path in bad_paths:
            # Find the result that contains this path
            result = None
            for r in results:
                if r.get("path") == path or (r.get("paths") and path in r["paths"]):
                    result = r
                    break
            if result:
                status = "ERROR" if result["error"] else "BLANK"
                print(f"- {path}: {status}")
                if result["error"]:
                    print(f"  Error: {result['error']}")
    
    print(f"\nDetailed results saved to:")
    print(f"- JSON: {os.path.join(output_dir, 'results.json')}")
    print(f"- CSV: {os.path.join(output_dir, 'results.csv')}")
    print(f"- Bad paths: {os.path.join(output_dir, 'bad_paths.txt')}")
    print("=======================\n")


# ------------------ Main ------------------
def main():
    """Main entry point for the VAE diagnostic script.
    
    Parses command-line arguments, sets up logging, run diagnostics on the VAE
    decoder submodules, and saves the results.
    """
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run diagnostic tests on the VAE decoder submodules of a diffusion pipeline."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="vae_diagnostic_output", 
        help="Directory to save diagnostic results and outputs (default: 'vae_diagnostic_output')."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="hpu", 
        choices=["hpu", "cpu"], 
        help="Device to run the pipeline on ('hpu' or 'cpu', default: 'hpu')."
    )
    parser.add_argument(
        "--filter", 
        type=str, 
        default="all", 
        choices=["all", "leaf", "non-leaf"], 
        help="Type of submodules to test ('all', 'leaf', or 'non-leaf', default: 'all')."
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="single", 
        choices=["single", "compile_except"], 
        help="Compilation mode ('single' or 'compile_except', default: 'single')."
    )
    parser.add_argument(
        "--exclude_path", 
        type=str, 
        help="Path to exclude in compile_except mode."
    )
    parser.add_argument(
        "--test_paths",
        type=str,
        help="Specific submodule paths to test. Can be a file path containing paths (one per line) or a comma-separated list of paths."
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save generated images for each test."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Name of the pretrained model to use."
    )
    parser.add_argument(
        "--gaudi_config",
        type=str,
        default="Habana/stable-diffusion",
        help="Path to Gaudi configuration file (default: 'Habana/stable-diffusion')."
    )
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        help="Enable TensorBoard logging."
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights and Biases logging."
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="vae_diagnostic", 
        help="Weights and Biases project name (default: 'vae_diagnostic')."
    )
    parser.add_argument(
        "--wandb_run", 
        type=str, 
        default="run_vae_test", 
        help="Weights and Biases run name (default: 'run_vae_test')."
    )
    parser.add_argument(
        "--list_submodules", 
        action="store_true", 
        help="List VAE decoder submodules hierarchically and save to a file."
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Set up TensorBoard writer if enabled
    tb_writer = None
    if args.use_tensorboard:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output, "tensorboard"))

    # Set up Weights and Biases if enabled
    wandb_run = None
    if args.use_wandb and WANDB_AVAILABLE:
        wandb_run = wandb.init(project="vae-diagnostic", config=vars(args))

    # List submodules if requests
    if args.list_submodules:
        pipe = create_pipeline(args.model_name, args.gaudi_config, args.device)
        lines = list_submodules(pipe.vae.decoder)

        print("VAE Decoder Submodules:")
        for line in lines:
            print(line)

        submodules_list_path = os.path.join(args.output, "vae_submodules_list.txt")
        with open(submodules_list_path, "w") as f:
            f.writelines('\n'.join(lines))
        print(f"Submodule list written to: {submodules_list_path}")

    # Define output file paths
    json_path = os.path.join(args.output, "results.json")
    csv_path = os.path.join(args.output, "results.csv")
    bad_paths_path = os.path.join(args.output, "bad_paths.txt")

    # Load specific paths to test if specified
    paths_to_test = []
    if args.test_paths:
        if os.path.exists(args.test_paths):
            # If the argument is a file path, read paths from file
            with open(args.test_paths, "r") as f:
                paths_to_test = [line.strip() for line in f if line.strip()]
        else:
            # If not a file, treat as comma-separated list
            paths_to_test = [path.strip() for path in args.test_paths.split(",") if path.strip()]
        
        if not paths_to_test:
            print("No paths found to test.")
            return
        
        print(f"\nTesting {len(paths_to_test)} specific submodules:")
        for path in paths_to_test:
            print(f"- {path}")
        print()
        
        # Override mode to single and filter to all for specific path testing
        args.mode = "single"
        args.filter = "all"

    # Run diagnostics
    if paths_to_test:
        # Create a custom submodules list with only the specified paths
        pipe = create_pipeline(args.model_name, args.gaudi_config, args.device)
        all_submodules = list(get_all_submodules(pipe.vae.decoder))
        submodules = [(path, sub) for path, sub in all_submodules if path in paths_to_test]
        results, bad_paths = run_diagnostic(
            args.model_name,
            args.gaudi_config,
            args.device,
            args.mode,
            args.filter,
            args.exclude_path,
            args.save_images,
            args.output,
            tb_writer,
            wandb_run,
            submodules=submodules
        )
    elif args.mode == "compile_except" and args.exclude_path is None:
        # If in compile_except mode but no exclude_path specified, try to load from bad_paths.txt
        bad_paths_file = os.path.join(args.output, "bad_paths.txt")
        if os.path.exists(bad_paths_file):
            with open(bad_paths_file, "r") as f:
                exclude_paths = [line.strip() for line in f if line.strip()]
            if exclude_paths:
                print(f"Found {len(exclude_paths)} problematic paths to exclude")
                results, bad_paths = run_diagnostic(
                    args.model_name,
                    args.gaudi_config,
                    args.device,
                    args.mode,
                    args.filter,
                    exclude_paths,  # Pass the list of paths to exclude
                    args.save_images,
                    args.output,
                    tb_writer,
                    wandb_run
                )
            else:
                print("No problematic paths found in bad_paths.txt")
                return
        else:
            print("No exclude_path specified and bad_paths.txt not found")
            return
    else:
        results, bad_paths = run_diagnostic(
            args.model_name,
            args.gaudi_config,
            args.device,
            args.mode,
            args.filter,
            args.exclude_path,
            args.save_images,
            args.output,
            tb_writer,
            wandb_run
        )

    # Save results
    save_results(results, bad_paths, json_path, csv_path, bad_paths_path)

    # Print test summary
    print_test_summary(results, bad_paths, args.output)

    # Clean up
    if tb_writer:
        tb_writer.close()
    if wandb_run:
        wandb_run.finish()

    print(f"\nDiagnostic results saved to: {args.output}")
    print(f"Found {len(bad_paths)} problematic submodules.")


if __name__ == "__main__":
    main()
