"""Utility functions for working with images and tensors."""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T


def is_blank(tensor):
    """Check if an image tensor is blank (all zeros or all ones).

    This function supports both torch.Tensor and PIL.Image inputs, and handles
    different tensor dimensions (batch and single images). It validates the input
    and provides comprehensive error handling.

    Args:
        tensor (Union[torch.Tensor, PIL.Image]): The image tensor or PIL Image to check.

    Returns:
        bool: True if the image is blank, False otherwise.

    Raises:
        ValueError: If the input is invalid or has unexpected properties.
    """
    try:
        # Convert PIL Image to tensor if necessary
        if isinstance(tensor, Image.Image):
            tensor = T.ToTensor()(tensor)
            
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
            
        # Check if all values are the same
        return torch.allclose(tensor, tensor[0, 0, 0])
        
    except Exception as e:
        raise ValueError(f"Error checking if image is blank: {str(e)}")


def save_image_tensor(tensor, output_path):
    """Save an image tensor to a file.

    This function supports both torch.Tensor and PIL.Image inputs, automatically
    creates directories if they don't exist, and handles different tensor dimensions
    (batch and single images). It validates the input and provides comprehensive
    error handling.

    Args:
        tensor (Union[torch.Tensor, PIL.Image]): The image tensor or PIL Image to save.
        output_path (str): The path where the image should be saved.

    Raises:
        ValueError: If the input is invalid or has unexpected properties.
        IOError: If there are issues creating directories or saving the file.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert PIL Image to tensor if necessary
        if isinstance(tensor, Image.Image):
            tensor = T.ToTensor()(tensor)
            
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
            raise ValueError(f"Expected 3D tensor (C,H,W) or 4D tensor (B,C,H,W), got {tensor.dim()}")
            
        # Validate number of channels
        if tensor.size(0) not in [1, 3]:
            raise ValueError(f"Expected 1 or 3 channels, got {tensor.size(0)}")
            
        # Convert tensor to PIL Image and save
        if tensor.size(0) == 1:  # Grayscale
            tensor = tensor.repeat(3, 1, 1)  # Convert to RGB
        tensor = tensor.clamp(0, 1)  # Ensure values are in [0, 1]
        image = T.ToPILImage()(tensor)
        image.save(output_path)
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise IOError(f"Error saving image to {output_path}: {str(e)}")


def tensor_to_pil(tensor):
    """Convert a tensor to a PIL Image.

    This function handles different tensor dimensions (batch and single images)
    and validates the input. It provides comprehensive error handling.

    Args:
        tensor (torch.Tensor): The image tensor to convert.

    Returns:
        PIL.Image: The converted PIL Image.

    Raises:
        ValueError: If the input is invalid or has unexpected properties.
    """
    try:
        # Validate tensor type
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
            
        # Check if tensor is empty
        if tensor.numel() == 0:
            raise ValueError("Input tensor is empty")
            
        # Handle batch dimension
        if tensor.dim() == 4:  # Batch of images
            tensor = tensor[0]  # Take first image
        elif tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor (C,H,W) or 4D tensor (B,C,H,W), got {tensor.dim()}")
            
        # Validate number of channels
        if tensor.size(0) not in [1, 3]:
            raise ValueError(f"Expected 1 or 3 channels, got {tensor.size(0)}")
            
        # Convert grayscale to RGB if necessary
        if tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)
            
        # Ensure values are in [0, 1]
        tensor = tensor.clamp(0, 1)
        
        # Convert to PIL Image
        return T.ToPILImage()(tensor)
        
    except Exception as e:
        raise ValueError(f"Error converting tensor to PIL Image: {str(e)}")


def pil_to_tensor(image):
    """Convert a PIL Image to a tensor.

    This function validates the input and provides comprehensive error handling.

    Args:
        image (PIL.Image): The PIL Image to convert.

    Returns:
        torch.Tensor: The converted tensor.

    Raises:
        ValueError: If the input is invalid or has unexpected properties.
    """
    try:
        # Validate input type
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL.Image")
            
        # Check if image is empty
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError("Input image is empty")
            
        # Convert to tensor
        tensor = T.ToTensor()(image)
        
        # Validate number of channels
        if tensor.size(0) not in [1, 3]:
            raise ValueError(f"Expected 1 or 3 channels, got {tensor.size(0)}")
            
        return tensor
        
    except Exception as e:
        raise ValueError(f"Error converting PIL Image to tensor: {str(e)}")


def normalize_tensor(tensor, mean=None, std=None):
    """Normalize a tensor using mean and standard deviation.

    This function handles different tensor dimensions (batch and single images)
    and validates the input. It provides comprehensive error handling.

    Args:
        tensor (torch.Tensor): The tensor to normalize.
        mean (Union[float, list], optional): Mean values for each channel.
        std (Union[float, list], optional): Standard deviation values for each channel.

    Returns:
        torch.Tensor: The normalized tensor.

    Raises:
        ValueError: If the input is invalid or has unexpected properties.
    """
    try:
        # Validate tensor type
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
            
        # Check if tensor is empty
        if tensor.numel() == 0:
            raise ValueError("Input tensor is empty")
            
        # Handle batch dimension
        if tensor.dim() == 4:  # Batch of images
            tensor = tensor[0]  # Take first image
        elif tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor (C,H,W) or 4D tensor (B,C,H,W), got {tensor.dim()}")
            
        # Validate number of channels
        if tensor.size(0) not in [1, 3]:
            raise ValueError(f"Expected 1 or 3 channels, got {tensor.size(0)}")
            
        # Set default values if not provided
        if mean is None:
            mean = [0.485, 0.456, 0.406] if tensor.size(0) == 3 else [0.5]
        if std is None:
            std = [0.229, 0.224, 0.225] if tensor.size(0) == 3 else [0.5]
            
        # Convert to tensors
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        
        # Normalize
        return (tensor - mean) / std
        
    except Exception as e:
        raise ValueError(f"Error normalizing tensor: {str(e)}")


def denormalize_tensor(tensor, mean=None, std=None):
    """Denormalize a tensor using mean and standard deviation.

    This function handles different tensor dimensions (batch and single images)
    and validates the input. It provides comprehensive error handling.

    Args:
        tensor (torch.Tensor): The tensor to denormalize.
        mean (Union[float, list], optional): Mean values for each channel.
        std (Union[float, list], optional): Standard deviation values for each channel.

    Returns:
        torch.Tensor: The denormalized tensor.

    Raises:
        ValueError: If the input is invalid or has unexpected properties.
    """
    try:
        # Validate tensor type
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
            
        # Check if tensor is empty
        if tensor.numel() == 0:
            raise ValueError("Input tensor is empty")
            
        # Handle batch dimension
        if tensor.dim() == 4:  # Batch of images
            tensor = tensor[0]  # Take first image
        elif tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor (C,H,W) or 4D tensor (B,C,H,W), got {tensor.dim()}")
            
        # Validate number of channels
        if tensor.size(0) not in [1, 3]:
            raise ValueError(f"Expected 1 or 3 channels, got {tensor.size(0)}")
            
        # Set default values if not provided
        if mean is None:
            mean = [0.485, 0.456, 0.406] if tensor.size(0) == 3 else [0.5]
        if std is None:
            std = [0.229, 0.224, 0.225] if tensor.size(0) == 3 else [0.5]
            
        # Convert to tensors
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        
        # Denormalize
        return tensor * std + mean
        
    except Exception as e:
        raise ValueError(f"Error denormalizing tensor: {str(e)}") 