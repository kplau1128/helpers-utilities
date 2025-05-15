"""Utility functions for working with PyTorch modules."""

import torch
import torch.nn as nn
from typing import Generator, Tuple, Any, Union, List


class CompiledWrapper(nn.Module):
    """A wrapper class to compile a PyTorch module using the HPU backend.

    This class encapsulates a PyTorch module, compiles it with the specified backend
    (hpu_backend), and preserves the original module type for diagnostic purposes.

    Args:
        module (nn.Module): The PyTorch module to be compiled.
    """
    def __init__(self, module):
        super().__init__()
        # Store the original module type for reference
        self.original_type = type(module)
        try:
            # Compile the module with the HPU backend for optimized execution
            self.compiled = torch.compile(module, backend="hpu_backend")
        except Exception as e:
            # If compilation fails, use the original module
            print(f"Warning: Failed to compile module {type(module).__name__}: {str(e)}")
            self.compiled = module

        # Copy all attributes and methods from the original module
        for name in dir(module):
            if not name.startswith('_') and name not in dir(self):
                try:
                    attr = getattr(module, name)
                    # Copy both callable and non-callable attributes
                    setattr(self, name, attr)
                except (AttributeError, TypeError):
                    continue

    def forward(self, *args, **kwargs):
        """Forward pass through the compiled module.

        Args:
            *args: Variable length argument list passed to the compiled module.
            **kwargs: Arbitrary keyword arguments passed to the compiled module.

        Returns:
            The output of the compiled module's forward pass.
        """
        return self.compiled(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Call the compiled module.

        Args:
            *args: Variable length argument list passed to the compiled module.
            **kwargs: Arbitrary keyword arguments passed to the compiled module.

        Returns:
            The output of the compiled module's call.
        """
        return self.compiled(*args, **kwargs)


def get_all_submodules(module: nn.Module, prefix: str = "") -> Generator[Tuple[str, nn.Module], None, None]:
    """Recursively retrieve all submodules of a PyTorch module with their hierarchical paths.

    This generator function traverses the module's hierarchy and yields tuples containing
    the path and submodule for each submodule encountered. It handles cases where some
    submodules might not have named_children() function.

    Args:
        module (nn.Module): The PyTorch module to traverse.
        prefix (str, optional): The prefix for the current module path. Defaults to "".

    Yields:
        tuple: A tuple containing the path (str) and submodule (nn.Module).
    """
    # First yield the module itself if it's not the root
    if prefix:
        yield prefix, module

    # Try different methods to get submodules
    try:
        # Method 1: Try named_children()
        for name, sub in module.named_children():
            path = f"{prefix}.{name}" if prefix else name
            yield from get_all_submodules(sub, path)
    except (AttributeError, TypeError):
        try:
            # Method 2: Try __dict__
            for name, sub in module.__dict__.items():
                if isinstance(sub, nn.Module):
                    path = f"{prefix}.{name}" if prefix else name
                    yield from get_all_submodules(sub, path)
        except (AttributeError, TypeError):
            # Method 3: Try direct attributes
            for name in dir(module):
                if not name.startswith('_'):  # Skip private attributes
                    try:
                        sub = getattr(module, name)
                        if isinstance(sub, nn.Module):
                            path = f"{prefix}.{name}" if prefix else name
                            yield from get_all_submodules(sub, path)
                    except (AttributeError, TypeError):
                        continue


def filter_submodules(modules: list, filter_type: str) -> list:
    """Filter a list of submodules based on the specified filter type.

    This function filters submodules based on whether they are leaf nodes (no children)
    or non-leaf nodes (have children). It handles cases where some submodules might
    not have children() function.

    Args:
        modules (list): List of tuples containing (path, submodule).
        filter_type (str): Type of filter to apply ("all", "leaf", or "non-leaf").

    Returns:
        list: Filtered list of (path, submodule) tuples.

    Raises:
        ValueError: If filter_type is not one of "all", "leaf", or "non-leaf".
    """
    if filter_type == "all":
        return modules
    elif filter_type == "leaf":
        return [(path, module) for path, module in modules if is_leaf_module(module)]
    elif filter_type == "non-leaf":
        return [(path, module) for path, module in modules if not is_leaf_module(module)]
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def is_leaf_module(module: nn.Module) -> bool:
    """Check if a module is a leaf module (has no children).

    Args:
        module (nn.Module): The module to check.

    Returns:
        bool: True if the module is a leaf, False otherwise.
    """
    try:
        # Try different methods to check for children
        try:
            # Method 1: Try children()
            return not any(module.children())
        except (AttributeError, TypeError):
            try:
                # Method 2: Try __dict__
                return not any(isinstance(v, nn.Module) for v in module.__dict__.values())
            except (AttributeError, TypeError):
                # Method 3: Try direct attributes
                return not any(
                    isinstance(getattr(module, name), nn.Module)
                    for name in dir(module)
                    if not name.startswith('_')
                )
    except Exception:
        # If all methods fail, assume it's a leaf
        return True


def is_wrappable_module(module: nn.Module) -> bool:
    """Check if a module can be wrapped for compilation.

    Args:
        module (nn.Module): The PyTorch module to check.

    Returns:
        bool: True if the module can be wrapped, False otherwise.
    """
    try:
        # Skip non-modules and already wrapped modules
        if not isinstance(module, nn.Module) or isinstance(module, CompiledWrapper):
            return False

        # Skip certain module types
        non_wrappable_types = (nn.ModuleList, nn.ModuleDict, nn.Sequential)
        if isinstance(module, non_wrappable_types):
            return False

        # Check for forward method
        if not hasattr(module, 'forward') or not callable(getattr(module, 'forward')):
            return False

        # Try to access basic module attributes
        _ = module.training
        _ = module._parameters
        return True
    except Exception:
        return False


def apply_compile_to_path(module: nn.Module, target_path: str, prefix: str = ""):
    """Apply compilation to a specific submodule at the given path.

    Args:
        module (nn.Module): The root PyTorch module to traverse.
        target_path (str): The path to the target submodule.
        prefix (str, optional): The current path prefix. Defaults to "".

    Returns:
        nn.Module: The module with compilation applied to the target submodule.
    """
    # First check if this is the target module
    if prefix == target_path and is_wrappable_module(module):
        try:
            return CompiledWrapper(module)
        except Exception as e:
            print(f"Warning: Failed to compile module {type(module).__name__}: {str(e)}")
            return module

    def try_compile_submodule(sub, path):
        """Helper function to try compiling a submodule."""
        if path == target_path and is_wrappable_module(sub):
            try:
                return CompiledWrapper(sub)
            except Exception as e:
                print(f"Warning: Failed to compile module {path}: {str(e)}")
        return sub

    # Try different methods to find and compile the target
    try:
        # Method 1: Try named_children()
        for name, sub in module.named_children():
            path = f"{prefix}.{name}" if prefix else name
            compiled_sub = try_compile_submodule(sub, path)
            if compiled_sub is not sub:
                setattr(module, name, compiled_sub)
            else:
                apply_compile_to_path(sub, target_path, path)
    except (AttributeError, TypeError):
        try:
            # Method 2: Try __dict__
            for name, sub in module.__dict__.items():
                if isinstance(sub, nn.Module):
                    path = f"{prefix}.{name}" if prefix else name
                    compiled_sub = try_compile_submodule(sub, path)
                    if compiled_sub is not sub:
                        setattr(module, name, compiled_sub)
                    else:
                        apply_compile_to_path(sub, target_path, path)
        except (AttributeError, TypeError):
            # Method 3: Try direct attributes
            for name in dir(module):
                if not name.startswith('_'):
                    try:
                        sub = getattr(module, name)
                        if isinstance(sub, nn.Module):
                            path = f"{prefix}.{name}" if prefix else name
                            compiled_sub = try_compile_submodule(sub, path)
                            if compiled_sub is not sub:
                                setattr(module, name, compiled_sub)
                            else:
                                apply_compile_to_path(sub, target_path, path)
                    except (AttributeError, TypeError):
                        continue

    return module


def apply_compile_except(module: nn.Module, skip_paths: Union[str, List[str]], prefix: str = ""):
    """Apply compilation to all submodules except the ones at the specified paths.

    Args:
        module (nn.Module): The root PyTorch module to traverse.
        skip_paths (Union[str, List[str]]): The path(s) of the submodule(s) to skip.
            Can be a single path string or a list of path strings.
        prefix (str, optional): The current path prefix. Defaults to "".

    Returns:
        nn.Module: The module with compilation applied to all non-skipped submodules.
    """
    # Convert single path to list for uniform handling
    if isinstance(skip_paths, str):
        skip_paths = [skip_paths]

    # Handle root module compilation if not in skip paths
    if prefix == "" and is_wrappable_module(module):
        try:
            module = CompiledWrapper(module)
        except Exception as e:
            print(f"Warning: Failed to compile root module {type(module).__name__}: {str(e)}")

    # Skip if this is one of the target paths
    if prefix in skip_paths:
        return module

    def try_compile_submodule(sub, path):
        """Helper function to try compiling a submodule."""
        if path not in skip_paths and is_wrappable_module(sub):
            try:
                return CompiledWrapper(sub)
            except Exception as e:
                print(f"Warning: Failed to compile module {path}: {str(e)}")
        return sub

    # Try different methods to find and compile modules
    try:
        # Method 1: Try named_children()
        for name, sub in module.named_children():
            path = f"{prefix}.{name}" if prefix else name
            compiled_sub = try_compile_submodule(sub, path)
            if compiled_sub is not sub:
                setattr(module, name, compiled_sub)
            apply_compile_except(sub, skip_paths, path)
    except (AttributeError, TypeError):
        try:
            # Method 2: Try __dict__
            for name, sub in module.__dict__.items():
                if isinstance(sub, nn.Module):
                    path = f"{prefix}.{name}" if prefix else name
                    compiled_sub = try_compile_submodule(sub, path)
                    if compiled_sub is not sub:
                        setattr(module, name, compiled_sub)
                    apply_compile_except(sub, skip_paths, path)
        except (AttributeError, TypeError):
            # Method 3: Try direct attributes
            for name in dir(module):
                if not name.startswith('_'):
                    try:
                        sub = getattr(module, name)
                        if isinstance(sub, nn.Module):
                            path = f"{prefix}.{name}" if prefix else name
                            compiled_sub = try_compile_submodule(sub, path)
                            if compiled_sub is not sub:
                                setattr(module, name, compiled_sub)
                            apply_compile_except(sub, skip_paths, path)
                    except (AttributeError, TypeError):
                        continue

    return module


def get_submodule_type(module: nn.Module, path: str) -> str:
    """Get the type of a submodule at the specified path.

    Args:
        module (nn.Module): The root module.
        path (str): The path to the submodule.

    Returns:
        str: The type name of the submodule.
    """
    try:
        # Try different methods to find the submodule
        try:
            # Method 1: Try named_modules()
            for name, sub in module.named_modules():
                if name == path:
                    return type(sub).__name__
        except (AttributeError, TypeError):
            # Method 2: Try direct attribute access
            parts = path.split('.')
            current = module
            for part in parts:
                current = getattr(current, part)
            return type(current).__name__
    except Exception:
        return "Unknown"


def get_submodule_orig_type(module: nn.Module, path: str) -> str:
    """Get the original type of a submodule at the specified path.

    Args:
        module (nn.Module): The root module.
        path (str): The path to the submodule.

    Returns:
        str: The original type name of the submodule.
    """
    try:
        # Try different methods to find the submodule
        try:
            # Method 1: Try named_modules()
            for name, sub in module.named_modules():
                if name == path:
                    if isinstance(sub, CompiledWrapper):
                        return sub.original_type.__name__
                    return type(sub).__name__
        except (AttributeError, TypeError):
            # Method 2: Try direct attribute access
            parts = path.split('.')
            current = module
            for part in parts:
                current = getattr(current, part)
            if isinstance(current, CompiledWrapper):
                return current.original_type.__name__
            return type(current).__name__
    except Exception:
        return "Unknown"


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
    try:
        # Try to get children using named_children()
        for name, module in model.named_children():
            # Construct the full path and indentation
            full_name = f"{prefix}.{name}" if prefix else name
            indent = '   ' * depth
            type_str = f"{type(module).__module__}.{type(module).__name__}"
            lines.append(f"{indent}- {full_name}: {type_str}")
            # Recursively list submodules
            lines.extend(list_submodules(module, full_name, depth + 1))
    except (AttributeError, TypeError):
        # If named_children() is not available, try to get children using __dict__
        try:
            for name, module in model.__dict__.items():
                if isinstance(module, nn.Module):
                    # Construct the full path and indentation
                    full_name = f"{prefix}.{name}" if prefix else name
                    indent = '   ' * depth
                    type_str = f"{type(module).__module__}.{type(module).__name__}"
                    lines.append(f"{indent}- {full_name}: {type_str}")
                    # Recursively list submodules
                    lines.extend(list_submodules(module, full_name, depth + 1))
        except (AttributeError, TypeError):
            # If neither method works, just list the module itself
            if prefix:  # Only list if it's not the root module
                indent = '   ' * depth
                type_str = f"{type(model).__module__}.{type(model).__name__}"
                lines.append(f"{indent}- {prefix}: {type_str}")
    return lines 