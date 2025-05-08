# Helpers-Utilities

This repository contains a collection of tools and utilities designed to assist with various tasks, including diagnostics for machine learning models and pipelines.

## Table of Contents

- [Helpers-Utilities](#helpers-utilities)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
    - [Files](#files)
  - [Features](#features)
  - [VAE Diagnostic Tool](#vae-diagnostic-tool)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Project Structure

```bash
helpers-utilities/
 ├── utility_diagnostic/
 │   └── vae/
 │       ├── vae_diagnostic.py
 │       └── README.md
 └── README.md
```

### Files

- **`utility_diagnostic/vae/vae_diagnostic.py`**: A Python script for running diagnostics on the VAE decoder submodules of a diffusion pipeline. It supports features like submodule compilation, image generation, and logging to TensorBoard or Weights and Biases.

## Features

- **Submodule Diagnostics**: Test the effect of compiling individual submodules or all submodules except a specified path.
- **Image Generation**: Generate images using a diffusion pipeline and check for blank outputs or errors.
- **Logging**: Log results to TensorBoard or Weights and Biases for better visualization.
- **Submodule Listing**: List all submodules of the VAE decoder hierarchically.

## VAE Diagnostic Tool

For detailed documentation about the VAE Diagnostic Tool, including features, usage, and best practices, please refer to the [VAE Diagnostic Tool Documentation](utility_diagnostic/vae/README.md).

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

## Usage

### Command-Line Arguments

Run the script with the following options:
```bash
python utility_diagnostic/vae/vae_diagnostic.py --help
```

### Example

To run diagnostics on all submodules and save images:
```bash
python utility_diagnostic/vae/vae_diagnostic.py --device hpu --filter all --mode single --save_images
```

To list all submodules of the VAE decoder:
```bash
python utility_diagnostic/vae/vae_diagnostic.py --list_submodules
```

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

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- **PyTorch**: For providing the deep learning framework used in this project.
- **Weights and Biases (wandb)**: For optional logging and visualization support.
- **TensorBoard**: For enabling performance monitoring and visualization.
- **Contributors**: Thanks to all contributors who have helped improve this project.
