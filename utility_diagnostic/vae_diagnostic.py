import torch
import torch.nn as nn
import argparse
import json
import csv
import os
import copy
from tqdm import tqdm
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


def is_blank(image_tensor, threshold=1e-4):
    """Check if an image tensor is blank based on its standard deviation.

    An image is considered blank if its standard deviation is below the threshold,
    indicatin low variation in pixel value.

    Args:
        image_tensor (torch.Tensor): The image tensor to check.
        threshold (float, optional): The standard deviation threshold. Defaults to 1e-4.

    Returns:
        bool: True if the image is blank, False otherwise.
    """
    #return image_tensor.mean().item() < threshold
    return image_tensor.std().item() < threshold


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
    including their paths and types.

    Args:
        model (nn.Module): The PyTorch module to traverse.
        prefix (str, optional): The current path prefix, Defaults to "".
        depth (int, optional): The current depth in the beirarchy. Defaults to 0.

    Returns:
        list: A list of strings decribing the submodules.
    """
    lines = []
    for name, module in model.named_children():
        # Construct the full path and indentation
        full_name = f"{prefix}.{name}" if prefix else name
        indent = '   ' * depth
        type_str = f"{type(module).__module__}.{type(module).__name__}"
        lines.append(f"{indent}- {full_name}: {type_str}")
        # Recursively list submodules
        lines.extend(list_submodules(module, full_name, depth + 1))
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
                   save_images, output_dir, tb_writer, wandb_run):
    """Run diagnostic tests on the VAE decoder submodules of a diffusion pipeline.
    
    The function tests the effect of compiling individual submodules or all submodules
    except a specified path, generating images and checking for blank outputs or errors.
    
    Args:
        model_name (str): The name of the pretrained model.
        gaudi_config (str): Configuration for Gaudi pipeline.
        device (str): The device to run the pipeline on ("hpu" or "cpu").
        mode (str): Compilation mode {"single", or "compile_except").
        filter_type (str): Type of submodules to test ("all", "leaf", or "non-leaf").
        exclude_path (str): Path to exclude in compile_except mode.
        save_images (bool): Whether to save generated images.
        output_dir (str): Directory to save generated images.
        tb_writer (SummaryWriter): TensorBoard writer for logging.
        wandb_run (wandb.Run): Weights and Biases run for logging.
        
    Returns:
        tuple: A tuple containing:
            - results (list): List of diagnostic results for each test.
            - bad_paths (list): List of paths that produced blank images or errors.
    """
    # Initialize the diffusion pipeline
    pipe = create_pipeline(model_name, gaudi_config, device)
    # Store the original VAE decoder for resetting between tests
    vae_decoder_orig = pipe.vae.decoder
    # Get all submodules of VAE decoder
    all_submodules = list(get_all_submodules(pipe.vae.decoder))
    # Filter submodules based on the specified type
    submodules = filter_submodules(all_submodules, filter_type)
    if save_images:
        # Create output directory for saving images
        os.makedirs(output_dir, exist_ok=True)

    results = []
    bad_paths = []
    # Define test prompt and infernecesteps
    prompt = "A picture of a dog in a bucket"
    num_inference_steps=25

    if mode == "compile_except":
        # Test compilation of all submodules except the excluded path
        print(f"[MODE] Compiling all except: {exclude_path}")
        pipe.vae.decoder = copy.deepcopy(vae_decoder_orig).to(device)
        try:
            apply_compile_except(pipe.vae.decoder, exclude_path)
            # Generate an image using the pipeline
            image = pipe(prompt=prompt, num_inference_steps=num_inference_steps).images[0]
            img_tensor = ToTensor()(image)
            blank = is_blank(img_tensor, 0.05)
            result = {
                "mode": "compile_except",
                "path": exclude_path,
                "blank": blank,
                "mean": img_tensor.mean().item(),
                "std": img_tensor.std().item(),
                "error": ""
            }
            if blank or result["mean"] == 0:
                bad_paths.append(exclude_path)
            if save_images:
                status = "BLANK" if blank else "OK"
                save_image_tensor(image, os.path.join(output_dir, f"except_{exclude_path.replace('.', '_')}_{status}.png"))
            if tb_writer:
                # Log metrics to TensorBoard
                tb_writer.add_scalar("mean", result["mean"], f"except_{exclude_path}")
                tb_writer.add_scalar("std", result["std"], f"except_{exclude_path}")
            if wandb_run:
                # Log metrics to Weights and Biases
                wandb_run.log({f"{exclude_path}/mean": result["mean"],
                               f"{exclude_path}/std": result["std"],
                               f"{exclude_path}/blank": blank})
        except Exception as e:
            err_msg = str(e)
            if "is not iterable" in str(e) and "CompiledWrapper" in str(e):
                orig_type = get_submodule_orig_type(pipe.vae.decoder, exclude_path)
                err_msg += f" ({orig_type.__module__}.{orig_type.__name__})"
            result = {
                "mode": "compile_except",
                "path": exclude_path,
                "blank": True,
                "mean": 0,
                "std": 0,
                "error": err_msg
            }
            bad_paths.append(exclude_path)
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
                blank = is_blank(img_tensor, 0.05)
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
                    save_image_tensor(image, os.path.join(output_dir, f"{path.replace('.', '_')}_{status}.png"))
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


# ------------------ Main ------------------
def main():
    """Main entry point for the VAE diagnostic script.
    
    Parses command-line arguments, sets up logging, run diagnostics on the VAE
    decoder submodules, and saves the results.
    """
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="vae_diagnostic_output")
    parser.add_argument("--device", type=str, default="hpu")
    parser.add_argument("--filter", type=str, default="all", choices=["all", "leaf", "non-leaf"])
    parser.add_argument("--mode", type=str, default="single", choices=["single", "compile_except"])
    parser.add_argument("--exclude_path", type=str, help="Path to exclude in compile_except mode")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="vae_diagnostic")
    parser.add_argument("--wandb_run", type=str, default="run_vae_test")
    parser.add_argument("--list-submodules", action="store_true", help="List VAE decoder submodules hierarchically")
    args = parser.parse_args()

    device = args.device
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Initialize TensorBoard writer if enabled
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output, "tensorboard")) if args.tensorboard else None

    # Initialize Weights and Biases if enabled and available
    wandb_run = None
    if args.wandb:
        if WANDB_AVAILABLE:
            wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run)
        else:
            print("[WARN] wandb not installed. Skipping wandb logging.")

    # Define model and configuration
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    gaudi_config = "Habana/stable-diffusion"
    
    # List submodules if requests
    if args.list_submodules:
        pipe = create_pipeline(model_name=model_name, gaudi_config=gaudi_config)
        lines = list_submodules(pipe.vae.decoder)

        print("VAE Decoder Submodules:")
        for line in lines:
            print(line)

        submodules_list_path = os.path.join(args.output, "vae_submodules_list.txt")
        with open(submodules_list_path, "w") as f:
            f.write("VAE Decoder Submodules:\n")
            f.write("\n".join(lines))
        print(f"Submodule list written to: {submodules_list_path}")

    # Run diagnostic tests
    print("[RUN] Diagnostic...")
    results, bad_paths = run_diagnostic(
        model_name=model_name,
        gaudi_config=gaudi_config,
        device=device,
        mode=args.mode,
        filter_type=args.filter,
        exclude_path=args.exclude_path,
        save_images=args.save_images,
        output_dir=os.path.join(args.output, "images") if args.save_images else None,
        tb_writer=tb_writer,
        wandb_run=wandb_run
    )

    # Clean up logging
    if tb_writer:
        tb_writer.flush()
        tb_writer.close()
    if wandb_run:
        wandb_run.finish()

    # Save diagnostic results
    json_path = os.path.join(args.output, "results.json")
    csv_path = os.path.join(args.output, "results.csv")
    bad_path = os.path.join(args.output, "bad_submodules.txt")
    save_results(results, bad_paths, json_path, csv_path, bad_path)

    # Print summary of results
    print("\n===== Summary =====")
    print(f"Total tested: {len(results)}")
    print(f"Blank outputs / errors: {len(bad_paths)}")
    print(f"Saved bad submodules to: {bad_path}")
    print("===================\n")


if __name__ == "__main__":
    main()
