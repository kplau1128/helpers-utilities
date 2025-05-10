# Helpers-Utilities

This repository contains a collection of tools and utilities designed to assist with various tasks, including diagnostics for machine learning models and pipelines.

## Table of Contents

- [Helpers-Utilities](#helpers-utilities)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
    - [Files](#files)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Contributing](#contributing)
  - [Acknowledgments](#acknowledgments)

## Project Structure

```bash
helpers-utilities/
 ├── utility_diagnostic/
 │   ├── utils/
 │   │   ├── __init__.py
 │   │   ├── arg_utils.py
 │   │   ├── image_utils.py
 │   │   ├── logging_utils.py
 │   │   ├── module_utils.py
 │   │   └── pipeline_utils.py
 │   ├── vae/
 │   │   ├── vae_diagnostic.py
 │   │   └── README.md
 │   └── pipeline/
 │       ├── __init__.py
 │       ├── pipeline_diagnostic.py
 │       └── README.md
 └── README.md
```

### Files

- **`utility_diagnostic/utils/`**: A collection of utility modules that provide common functionality:
  - `arg_utils.py`: Command-line argument parsing and validation
  - `image_utils.py`: Image processing and manipulation utilities
  - `logging_utils.py`: Logging configuration and management
  - `module_utils.py`: Module manipulation and inspection utilities
  - `pipeline_utils.py`: Pipeline-specific utility functions
- **`utility_diagnostic/vae/vae_diagnostic.py`**: A Python script for running diagnostics on the VAE decoder submodules of a diffusion pipeline. For detailed documentation, please refer to the [VAE Diagnostic Tool Documentation](utility_diagnostic/vae/README.md).
- **`utility_diagnostic/pipeline/pipeline_diagnostic.py`**: A Python script for running diagnostics on the entire Stable Diffusion pipeline submodules. For detailed documentation, please refer to the [Pipeline Diagnostic Tool Documentation](utility_diagnostic/pipeline/README.md).

## Features

- **VAE Submodule Diagnostics**: Test the effect of compiling individual VAE decoder submodules or all submodules except a specified path.
- **Pipeline Submodule Diagnostics**: Test the effect of compiling individual pipeline submodules or all submodules except a specified path.
- **Image Generation**: Generate images using a diffusion pipeline and check for blank outputs or errors.
- **Logging**: Log results to TensorBoard or Weights and Biases for better visualization.
- **Submodule Listing**: List all submodules of the VAE decoder or pipeline hierarchically.

## Requirements

- Python 3.8 or higher
- PyTorch
- TorchVision
- tqdm
- TensorBoard
- diffusers
- optimum-habana
- Optional: Weights and Biases (`wandb`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/helpers-utilities.git
   cd helpers-utilities
   ```
2. Install the required dependencies:
   ```bash
   pip install torch torchvision tqdm tensorboard diffusers optimum-habana
   ```
3. (Optional) Install Weights and Biases:
   ```bash
   pip install wandb
   ```

## Quick Start

### VAE Diagnostic Tool

To run diagnostics on the VAE decoder:

```bash
python utility_diagnostic/vae/vae_diagnostic.py --list_submodules
```

For more details, see the [VAE Diagnostic Tool Documentation](utility_diagnostic/vae/README.md).

### Pipeline Diagnostic Tool

To run diagnostics on the entire pipeline:

```bash
python utility_diagnostic/pipeline/pipeline_diagnostic.py --list_submodules
```

For more details, see the [Pipeline Diagnostic Tool Documentation](utility_diagnostic/pipeline/README.md).

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add a meaningful commit message"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## Acknowledgments

- **PyTorch**: For providing the deep learning framework used in this project.
- **Weights and Biases (wandb)**: For optional logging and visualization support.
- **TensorBoard**: For enabling performance monitoring and visualization.
- **Contributors**: Thanks to all contributors who have helped improve this project.
