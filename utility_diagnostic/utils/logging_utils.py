"""Utility functions for logging and monitoring."""

import os
import time
import logging
import warnings
from datetime import datetime
from typing import Optional, Dict, Any


def setup_logging(log_dir: str, name: str = "diagnostic") -> logging.Logger:
    """Set up logging configuration.

    This function configures logging to both file and console, with different
    log levels and formats. It handles directory creation and provides
    comprehensive error handling.

    Args:
        log_dir (str): Directory to store log files.
        name (str, optional): Name of the logger. Defaults to "diagnostic".

    Returns:
        logging.Logger: The configured logger.

    Raises:
        IOError: If there are issues creating directories or log files.
    """
    try:
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        logger.handlers = []

        # Create file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            raise IOError(f"Failed to create file handler: {str(e)}")

        # Create console handler
        try:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(levelname)s: %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        except Exception as e:
            raise IOError(f"Failed to create console handler: {str(e)}")

        return logger

    except Exception as e:
        if isinstance(e, IOError):
            raise
        raise IOError(f"Unexpected error setting up logging: {str(e)}")


def setup_tensorboard(log_dir: str) -> Optional[Any]:
    """Set up TensorBoard logging.

    This function initializes TensorBoard logging if the required packages are
    available. It provides comprehensive error handling.

    Args:
        log_dir (str): Directory to store TensorBoard logs.

    Returns:
        Optional[Any]: The TensorBoard SummaryWriter if successful, None otherwise.
    """
    try:
        # Try to import TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            warnings.warn("TensorBoard not available. Skipping TensorBoard setup.")
            return None

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Create SummaryWriter
        try:
            writer = SummaryWriter(log_dir=log_dir)
            return writer
        except Exception as e:
            warnings.warn(f"Failed to create TensorBoard writer: {str(e)}")
            return None

    except Exception as e:
        warnings.warn(f"Unexpected error setting up TensorBoard: {str(e)}")
        return None


def setup_wandb(project_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """Set up Weights & Biases logging.

    This function initializes Weights & Biases logging if the required packages
    are available. It provides comprehensive error handling.

    Args:
        project_name (str): Name of the W&B project.
        config (Optional[Dict[str, Any]], optional): Configuration to log.
            Defaults to None.

    Returns:
        Optional[Any]: The W&B run if successful, None otherwise.
    """
    try:
        # Try to import wandb
        try:
            import wandb
        except ImportError:
            warnings.warn("Weights & Biases not available. Skipping W&B setup.")
            return None

        # Initialize W&B
        try:
            run = wandb.init(
                project=project_name,
                config=config or {},
                reinit=True
            )
            return run
        except Exception as e:
            warnings.warn(f"Failed to initialize W&B: {str(e)}")
            return None

    except Exception as e:
        warnings.warn(f"Unexpected error setting up W&B: {str(e)}")
        return None


class Timer:
    """A context manager for timing code blocks.

    This class provides a context manager for timing code blocks and logging
    the elapsed time. It can be used with TensorBoard and W&B for monitoring.

    Args:
        name (str): Name of the timer.
        logger (Optional[logging.Logger], optional): Logger to use. Defaults to None.
        writer (Optional[Any], optional): TensorBoard writer. Defaults to None.
        wandb_run (Optional[Any], optional): W&B run. Defaults to None.
    """
    def __init__(
        self,
        name: str,
        logger: Optional[logging.Logger] = None,
        writer: Optional[Any] = None,
        wandb_run: Optional[Any] = None
    ):
        self.name = name
        self.logger = logger
        self.writer = writer
        self.wandb_run = wandb_run
        self.start_time = None
        self.elapsed_time = None

    def __enter__(self):
        """Start the timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and log the elapsed time."""
        self.elapsed_time = time.time() - self.start_time

        # Log to console/file
        if self.logger:
            self.logger.info(f"{self.name}: {self.elapsed_time:.2f} seconds")

        # Log to TensorBoard
        if self.writer:
            try:
                self.writer.add_scalar(f"time/{self.name}", self.elapsed_time)
            except Exception as e:
                warnings.warn(f"Failed to log time to TensorBoard: {str(e)}")

        # Log to W&B
        if self.wandb_run:
            try:
                self.wandb_run.log({f"time/{self.name}": self.elapsed_time})
            except Exception as e:
                warnings.warn(f"Failed to log time to W&B: {str(e)}")


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    logger: Optional[logging.Logger] = None,
    writer: Optional[Any] = None,
    wandb_run: Optional[Any] = None
) -> None:
    """Log metrics to various backends.

    This function logs metrics to the console/file, TensorBoard, and W&B if
    available. It provides comprehensive error handling.

    Args:
        metrics (Dict[str, float]): Dictionary of metric names and values.
        step (int): Current step number.
        logger (Optional[logging.Logger], optional): Logger to use. Defaults to None.
        writer (Optional[Any], optional): TensorBoard writer. Defaults to None.
        wandb_run (Optional[Any], optional): W&B run. Defaults to None.
    """
    try:
        # Log to console/file
        if logger:
            for name, value in metrics.items():
                logger.info(f"{name}: {value:.4f}")

        # Log to TensorBoard
        if writer:
            try:
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
            except Exception as e:
                warnings.warn(f"Failed to log metrics to TensorBoard: {str(e)}")

        # Log to W&B
        if wandb_run:
            try:
                wandb_run.log(metrics, step=step)
            except Exception as e:
                warnings.warn(f"Failed to log metrics to W&B: {str(e)}")

    except Exception as e:
        warnings.warn(f"Unexpected error logging metrics: {str(e)}")


def cleanup_logging(
    logger: Optional[logging.Logger] = None,
    writer: Optional[Any] = None,
    wandb_run: Optional[Any] = None
) -> None:
    """Clean up logging resources.

    This function closes and cleans up logging resources. It provides
    comprehensive error handling.

    Args:
        logger (Optional[logging.Logger], optional): Logger to clean up.
            Defaults to None.
        writer (Optional[Any], optional): TensorBoard writer to clean up.
            Defaults to None.
        wandb_run (Optional[Any], optional): W&B run to clean up.
            Defaults to None.
    """
    try:
        # Clean up logger
        if logger:
            try:
                for handler in logger.handlers:
                    handler.close()
                logger.handlers = []
            except Exception as e:
                warnings.warn(f"Failed to clean up logger: {str(e)}")

        # Clean up TensorBoard writer
        if writer:
            try:
                writer.close()
            except Exception as e:
                warnings.warn(f"Failed to clean up TensorBoard writer: {str(e)}")

        # Clean up W&B run
        if wandb_run:
            try:
                wandb_run.finish()
            except Exception as e:
                warnings.warn(f"Failed to clean up W&B run: {str(e)}")

    except Exception as e:
        warnings.warn(f"Unexpected error cleaning up logging: {str(e)}")