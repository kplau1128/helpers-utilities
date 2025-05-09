"""Utility functions for parsing command-line arguments."""

import os
import argparse
import warnings
from typing import Optional, Dict, Any


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the diagnostic tool.

    This function defines and parses all command-line arguments for the diagnostic
    tool, including model configuration, device settings, and output options.
    It provides comprehensive error handling and validation.

    Returns:
        argparse.Namespace: The parsed arguments.

    Raises:
        ValueError: If the arguments are invalid or incompatible.
    """
    try:
        # Create argument parser
        parser = argparse.ArgumentParser(
            description="Diagnostic tool for diffusion models",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Model configuration
        parser.add_argument(
            "--model_name",
            type=str,
            required=True,
            help="Name or path of the model to test"
        )
        parser.add_argument(
            "--gaudi_config",
            type=str,
            help="Gaudi configuration for HPU devices"
        )
        
        # Device settings
        parser.add_argument(
            "--device",
            type=str,
            choices=["hpu", "cuda", "cpu"],
            default="cpu",
            help="Device to run the model on"
        )
        
        # Test configuration
        parser.add_argument(
            "--mode",
            type=str,
            choices=["compile_except", "single"],
            required=True,
            help="Test mode to use"
        )
        parser.add_argument(
            "--filter_type",
            type=str,
            choices=["all", "leaf", "non-leaf"],
            default="all",
            help="Type of modules to test"
        )
        parser.add_argument(
            "--exclude_path",
            type=str,
            help="Path to exclude from compilation (for compile_except mode)"
        )
        parser.add_argument(
            "--bad_paths_file",
            type=str,
            help="File containing paths to exclude (for compile_except mode)"
        )
        
        # Output settings
        parser.add_argument(
            "--output_dir",
            type=str,
            default="diagnostic_output",
            help="Directory to save output files"
        )
        parser.add_argument(
            "--log_dir",
            type=str,
            default="logs",
            help="Directory to save log files"
        )
        parser.add_argument(
            "--tensorboard_dir",
            type=str,
            default="tensorboard",
            help="Directory to save TensorBoard logs"
        )
        
        # Logging options
        parser.add_argument(
            "--use_tensorboard",
            action="store_true",
            help="Enable TensorBoard logging"
        )
        parser.add_argument(
            "--use_wandb",
            action="store_true",
            help="Enable Weights & Biases logging"
        )
        parser.add_argument(
            "--wandb_project",
            type=str,
            default="model-diagnostic",
            help="Weights & Biases project name"
        )
        
        # Parse arguments
        try:
            args = parser.parse_args()
        except Exception as e:
            raise ValueError(f"Failed to parse arguments: {str(e)}")
            
        # Validate argument combinations
        if args.mode == "compile_except":
            if not args.exclude_path and not args.bad_paths_file:
                raise ValueError(
                    "compile_except mode requires either --exclude_path or --bad_paths_file"
                )
                
        # Validate paths
        if args.bad_paths_file and not os.path.exists(args.bad_paths_file):
            raise ValueError(f"Bad paths file not found: {args.bad_paths_file}")
            
        # Set default Gaudi config for HPU
        if args.device == "hpu" and not args.gaudi_config:
            warnings.warn(
                "No Gaudi config provided for HPU device, using default: Habana/stable-diffusion"
            )
            args.gaudi_config = "Habana/stable-diffusion"
            
        return args
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Unexpected error parsing arguments: {str(e)}")


def get_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert parsed arguments to a configuration dictionary.

    This function converts the parsed arguments to a dictionary that can be
    used for logging and configuration. It provides comprehensive error handling.

    Args:
        args (argparse.Namespace): The parsed arguments.

    Returns:
        Dict[str, Any]: The configuration dictionary.

    Raises:
        ValueError: If the arguments are invalid.
    """
    try:
        # Create configuration dictionary
        config = {
            "model_name": args.model_name,
            "device": args.device,
            "mode": args.mode,
            "filter_type": args.filter_type,
            "output_dir": args.output_dir,
            "log_dir": args.log_dir,
            "tensorboard_dir": args.tensorboard_dir,
            "use_tensorboard": args.use_tensorboard,
            "use_wandb": args.use_wandb,
            "wandb_project": args.wandb_project
        }
        
        # Add optional arguments if present
        if args.gaudi_config:
            config["gaudi_config"] = args.gaudi_config
        if args.exclude_path:
            config["exclude_path"] = args.exclude_path
        if args.bad_paths_file:
            config["bad_paths_file"] = args.bad_paths_file
            
        return config
        
    except Exception as e:
        raise ValueError(f"Failed to create configuration: {str(e)}")


def validate_args(args: argparse.Namespace) -> None:
    """Validate parsed arguments.

    This function performs additional validation on the parsed arguments,
    checking for compatibility and required dependencies. It provides
    comprehensive error handling.

    Args:
        args (argparse.Namespace): The parsed arguments.

    Raises:
        ValueError: If the arguments are invalid or incompatible.
    """
    try:
        # Check TensorBoard availability
        if args.use_tensorboard:
            try:
                import torch.utils.tensorboard
            except ImportError:
                raise ValueError(
                    "TensorBoard requested but not available. "
                    "Install with: pip install tensorboard"
                )
                
        # Check W&B availability
        if args.use_wandb:
            try:
                import wandb
            except ImportError:
                raise ValueError(
                    "Weights & Biases requested but not available. "
                    "Install with: pip install wandb"
                )
                
        # Check HPU availability
        if args.device == "hpu":
            try:
                import habana_frameworks.torch.core
            except ImportError:
                raise ValueError(
                    "HPU device requested but Habana frameworks not available. "
                    "Install with: pip install habana-frameworks"
                )
                
        # Check CUDA availability
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA device requested but not available. "
                    "Check your CUDA installation."
                )
                
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Unexpected error validating arguments: {str(e)}") 