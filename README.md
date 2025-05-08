# Helpers-Utilities

This repository contains a collection of tools and utilities designed to assist with various tasks, including diagnostics for machine learning models and pipelines.

## Table of Contents

- [Helpers-Utilities](#helpers-utilities)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
    - [Files](#files)
  - [Features](#features)
  - [VAE Diagnostic Tool](#vae-diagnostic-tool)
    - [Key Features](#key-features)
    - [Test Parameters](#test-parameters)
    - [Command-Line Arguments](#command-line-arguments)
    - [Usage Examples](#usage-examples)
    - [Output Structure](#output-structure)
    - [Interpreting Results](#interpreting-results)
    - [Test Summary](#test-summary)
    - [Best Practices](#best-practices)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Command-Line Arguments](#command-line-arguments)
    - [Example](#example)
  - [Output](#output)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Project Structure

```bash
helpers-utilities/
 ├── utility_diagnostic/
 │   └── vae_diagnostic.py
 └── README.md
```

### Files

- **`utility_diagnostic/vae_diagnostic.py`**: A Python script for running diagnostics on the VAE decoder submodules of a diffusion pipeline. It supports features like submodule compilation, image generation, and logging to TensorBoard or Weights and Biases.

## Features

- **Submodule Diagnostics**: Test the effect of compiling individual submodules or all submodules except a specified path.
- **Image Generation**: Generate images using a diffusion pipeline and check for blank outputs or errors.
- **Logging**: Log results to TensorBoard or Weights and Biases for better visualization.
- **Submodule Listing**: List all submodules of the VAE decoder hierarchically.

## VAE Diagnostic Tool

The VAE (Variational Autoencoder) diagnostic tool is designed to help diagnose and optimize the VAE decoder submodules in diffusion pipelines. It provides comprehensive testing capabilities for identifying problematic submodules and optimizing their compilation, with special support for HPU (Habana Processing Unit) devices.

### Key Features

- **HPU Optimization**: Specialized support for Habana Processing Units with optimized compilation
- **Compilation Testing**: Test individual submodules or all submodules except a specified path
- **Blank Image Detection**: Automatically detect and report blank or problematic image outputs
- **Performance Metrics**: Track mean and standard deviation of generated images
- **Flexible Logging**: Support for TensorBoard and Weights & Biases integration
- **Hierarchical Analysis**: Test different types of submodules (all, leaf, or non-leaf)
- **Module Type Preservation**: Maintains original module types for diagnostic purposes
- **Smart Module Filtering**: Intelligent handling of non-wrappable modules (ModuleList, ModuleDict, Sequential)
- **Deep Copy Testing**: Uses deep copy of VAE decoder for each test to ensure clean state
- **Progress Tracking**: Visual progress bar for submodule testing
- **Detailed Error Reporting**: Enhanced error messages with original module type information
- **Comprehensive Metrics**: Per-module tracking of image statistics and compilation status
- **Multi-path Testing**: Support for testing multiple excluded paths in compile_except mode

### Test Parameters

The diagnostic tool uses the following default parameters for testing:

- **Test Prompt**: "A picture of a dog in a bucket"
- **Inference Steps**: 25
- **Blank Image Threshold**: 0.05 (standard deviation)
- **Image Format**: PNG
- **Test Modes**: 
  - `single`: Test individual submodules
  - `compile_except`: Test all submodules except specified path(s)

### Compilation Modes

The tool supports two main compilation modes:

1. **Single Mode** (`--mode single`):
   - Tests each submodule individually
   - Compiles only the target submodule
   - Useful for identifying specific problematic modules
   - Provides detailed per-module metrics

2. **Compile Except Mode** (`--mode compile_except`):
   - Compiles all submodules except the specified path(s)
   - Supports both single path and multiple paths exclusion
   - Useful for testing module interactions
   - Helps identify modules that cause issues when compiled together
   - Can be used to incrementally build a working compilation configuration

#### Using Compile Except Mode

1. **Single Path Exclusion**:
   ```bash
   python utility_diagnostic/vae_diagnostic.py --mode compile_except --exclude_path "path.to.submodule"
   ```

2. **Multiple Path Exclusion**:
   ```bash
   python utility_diagnostic/vae_diagnostic.py --mode compile_except --exclude_path "path1,path2,path3"
   ```

3. **Using Bad Paths File**:
   ```bash
   python utility_diagnostic/vae_diagnostic.py --mode compile_except --retest_bad_paths "bad_paths.txt"
   ```

#### Compile Except Mode Results

When using compile_except mode, the results include:

1. **Metrics**:
   - Combined metrics for all compiled modules
   - Individual metrics for excluded modules
   - Interaction effects between modules

2. **Output Files**:
   - Images named with excluded path(s)
   - Metrics organized by excluded path
   - Error messages specific to module interactions

3. **Error Handling**:
   - Type compatibility checks for all modules
   - Hierarchical error tracking
   - Module interaction error detection

### Best Practices for Compile Except Mode

1. Start with single path exclusion to identify problematic modules
2. Use the bad paths file to track problematic modules
3. Incrementally add paths to exclusion list
4. Monitor module interactions in the results
5. Use TensorBoard to visualize interaction effects
6. Check error messages for module type compatibility
7. Use the CSV output to analyze module relationships
8. Save images to visually verify compilation effects

### Metrics and Logging

The tool provides comprehensive metrics tracking through multiple channels:

1. **TensorBoard Metrics**:
   - Per-module mean values
   - Per-module standard deviation
   - Hierarchical metric organization by module path

2. **Weights & Biases Metrics**:
   - Module-specific metrics under path-based namespaces
   - Mean and standard deviation tracking
   - Blank image detection status
   - Real-time metric visualization

3. **File-based Logging**:
   - JSON format for detailed analysis
   - CSV format for spreadsheet processing
   - Text file for quick reference of problematic modules

### Error Handling

The tool implements sophisticated error handling:

1. **Compilation Errors**:
   - Detailed error messages with module type information
   - Original module type preservation for debugging
   - Hierarchical error tracking for multi-path tests

2. **Image Generation Errors**:
   - Blank image detection with configurable threshold
   - Zero-mean output detection
   - Per-module error status tracking

3. **Module Type Errors**:
   - Automatic detection of non-wrappable modules
   - Type compatibility checking
   - Original type preservation for debugging

### Command-Line Arguments

The VAE diagnostic tool supports the following command-line arguments:

```bash
--output OUTPUT       Directory to save diagnostic results and outputs (default: 'vae_diagnostic_output')
--device DEVICE       Device to run the pipeline on ('hpu' or 'cpu', default: 'hpu')
--filter FILTER       Type of submodules to test ('all', 'leaf', or 'non-leaf', default: 'all')
--mode MODE           Compilation mode ('single' or 'compile_except', default: 'single')
--exclude_path PATH   Path to exclude in 'compile_except' mode
--save_images         Save generated images to the output directory
--tensorboard         Enable logging metrics to TensorBoard
--wandb              Enable logging metrics to Weights and Biases
--wandb_project NAME  Weights and Biases project name (default: 'vae_diagnostic')
--wandb_run NAME     Weights and Biases run name (default: 'run_vae_test')
--list-submodules    List VAE decoder submodules hierarchically and save to a file
--model_name NAME    Name of the pretrained model to use (default: 'stabilityai/stable-diffusion-xl-base-1.0')
--gaudi_config PATH  Path to Gaudi configuration file for HPU optimization
--retest_bad_paths   Path to a file containing previously identified problematic submodule paths to re-test
```

### Usage Examples

1. **Basic Diagnostic Run with HPU**:
   ```bash
   python utility_diagnostic/vae_diagnostic.py --device hpu --filter all --mode single
   ```

2. **Save Generated Images with Custom Model**:
   ```bash
   python utility_diagnostic/vae_diagnostic.py --device hpu --save_images --model_name "custom/model/path"
   ```

3. **Test All Except Specific Path with Gaudi Config**:
   ```bash
   python utility_diagnostic/vae_diagnostic.py --mode compile_except --exclude_path "path.to.submodule" --gaudi_config "path/to/config.json"
   ```

4. **Enable TensorBoard Logging**:
   ```bash
   python utility_diagnostic/vae_diagnostic.py --tensorboard
   ```

5. **Enable Weights & Biases Logging**:
   ```bash
   python utility_diagnostic/vae_diagnostic.py --wandb --wandb_project "my_project" --wandb_run "test_run"
   ```

6. **List All Submodules**:
   ```bash
   python utility_diagnostic/vae_diagnostic.py --list-submodules
   ```

7. **Re-test Previously Identified Bad Paths**:
   ```bash
   python utility_diagnostic/vae_diagnostic.py --retest_bad_paths "path/to/bad_paths.txt"
   ```

### Output Structure

The tool generates the following outputs in the specified output directory:

```
output_directory/
├── images/                  # Generated images (if --save_images enabled)
│   ├── path_to_module_OK.png
│   └── path_to_module_BLANK.png
├── tensorboard/            # TensorBoard logs (if --tensorboard enabled)
│   └── metrics/           # Hierarchical metric organization
├── results.json           # Detailed results in JSON format
├── results.csv            # Results in CSV format
├── bad_submodules.txt     # List of problematic submodules
└── vae_submodules_list.txt # List of all submodules (if --list-submodules enabled)
```

### Interpreting Results

- **results.json/csv**: Contains detailed information about each test, including:
  - Module path(s)
  - Test mode (single/compile_except)
  - Image statistics (mean, std)
  - Blank image detection
  - Error messages with module type information
  - Original module type (for wrapped modules)
  - Test status (OK/BLANK/ERROR)

- **bad_submodules.txt**: Lists all submodules that either:
  - Produced blank images
  - Generated errors during compilation
  - Had zero mean output
  - Failed to compile due to module type incompatibility
  - Caused errors in multi-path testing

### Test Summary

The tool provides a detailed test summary including:
- Total number of tests run
- Number of successful tests
- Number of blank outputs
- Number of errors
- List of problematic submodules with their status and error messages
- Paths to detailed result files
- Hierarchical organization of test results

### Best Practices

1. Start with `--list-submodules` to understand the VAE decoder structure
2. Use `--filter leaf` for initial testing of leaf modules
3. Enable `--save_images` to visually inspect problematic outputs
4. Use TensorBoard or Weights & Biases for detailed performance analysis
5. Test in `compile_except` mode to identify specific problematic modules
6. For HPU devices, always provide a valid `--gaudi_config` file
7. Use `--retest_bad_paths` to efficiently re-test previously identified problematic modules
8. Monitor the original module types in the results to identify compilation compatibility issues
9. Check the test summary for a quick overview of test results
10. Use the CSV output for detailed analysis in spreadsheet software
11. Enable both TensorBoard and Weights & Biases for comprehensive metric tracking
12. Review error messages with module type information for debugging

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
python [vae_diagnostic.py](http://_vscodecontentref_/3) --help
```

### Example

To run diagnostics on all submodules and save images:
```bash
python [vae_diagnostic.py](http://_vscodecontentref_/4) --device hpu --filter all --mode single --save_images
```

To list all submodules of the VAE decoder:
```bash
python [vae_diagnostic.py](http://_vscodecontentref_/5) --list-submodules
```

## Output

The script generates the following outputs:

- Images: Saved in the specified output directory if --save_images is enabled.
- Results: Diagnostic results saved as JSON and CSV files.
- Bad Submodules: A text file listing submodules that produced blank images or errors.

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
