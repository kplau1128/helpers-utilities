"""Utility functions for the diagnostic tool.

This package contains various utility functions used by the diagnostic tool,
organized into modules based on their functionality:

- module_utils.py: Functions for working with PyTorch modules
- image_utils.py: Functions for working with images and tensors
- pipeline_utils.py: Functions for working with diffusion pipelines
- logging_utils.py: Functions for logging and monitoring
- arg_utils.py: Functions for parsing command-line arguments
"""

from .module_utils import (
    CompiledWrapper,
    get_all_submodules,
    filter_submodules,
    is_wrappable_module,
    apply_compile_to_path,
    apply_compile_except,
    get_submodule_type,
    get_submodule_orig_type,
    list_submodules
)

from .image_utils import (
    is_blank,
    save_image_tensor,
    tensor_to_pil,
    pil_to_tensor,
    normalize_tensor,
    denormalize_tensor
)

from .pipeline_utils import (
    create_pipeline,
    save_results,
    print_test_summary,
    list_pipeline_submodules
)

from .logging_utils import (
    setup_logging,
    setup_tensorboard,
    setup_wandb,
    Timer,
    log_metrics,
    cleanup_logging
)

from .arg_utils import (
    parse_arguments,
    get_config_from_args,
    validate_args
)

__all__ = [
    # Module utilities
    "CompiledWrapper",
    "get_all_submodules",
    "filter_submodules",
    "is_wrappable_module",
    "apply_compile_to_path",
    "apply_compile_except",
    "get_submodule_type",
    "get_submodule_orig_type",
    "list_submodules",
    
    # Image utilities
    "is_blank",
    "save_image_tensor",
    "tensor_to_pil",
    "pil_to_tensor",
    "normalize_tensor",
    "denormalize_tensor",
    
    # Pipeline utilities
    "create_pipeline",
    "save_results",
    "print_test_summary",
    "list_pipeline_submodules",
    
    # Logging utilities
    "setup_logging",
    "setup_tensorboard",
    "setup_wandb",
    "Timer",
    "log_metrics",
    "cleanup_logging",
    
    # Argument utilities
    "parse_arguments",
    "get_config_from_args",
    "validate_args"
] 