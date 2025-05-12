# VAE Diagnostic Tool

The VAE (Variational Autoencoder) diagnostic tool is designed to help diagnose and optimize the VAE decoder submodules in diffusion pipelines. It provides comprehensive testing capabilities for identifying problematic submodules and optimizing their compilation, with special support for HPU (Habana Processing Unit) devices.

## Table of Contents

- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Test Parameters](#test-parameters)
- [Compilation Modes](#compilation-modes)
  - [Using Compile Except Mode](#using-compile-except-mode)
  - [Compile Except Mode Results](#compile-except-mode-results)
- [Best Practices for Compile Except Mode](#best-practices-for-compile-except-mode)
- [Metrics and Logging](#metrics-and-logging)
- [Error Handling](#error-handling)
- [Command-Line Arguments](#command-line-arguments)
- [Testing Specific Paths](#testing-specific-paths)
- [Path Testing Behavior](#path-testing-behavior)
- [Retesting Workflow](#retesting-workflow)
- [Best Practices for Path Testing](#best-practices-for-path-testing)
- [Usage Examples](#usage-examples)
- [Output Structure](#output-structure)
- [Interpreting Results](#interpreting-results)
- [Test Summary](#test-summary)
- [Best Practices](#best-practices)

## Key Features

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

1. Install the required dependencies:
   ```bash
   pip install torch torchvision tqdm tensorboard diffusers optimum-habana
   ```
2. (Optional) Install Weights and Biases:
   ```bash
   pip install wandb
   ```

## Usage

### Command-Line Arguments

Run the script with the following options:
```bash
python vae_diagnostic.py --help
```

### Example

To run diagnostics on all submodules and save images:
```bash
python vae_diagnostic.py --device hpu --filter all --mode single --save_images
```

To list all submodules of the VAE decoder:
```bash
python vae_diagnostic.py --list_submodules
```

## Test Parameters

The diagnostic tool uses the following default parameters for testing:

- **Test Prompt**: "A picture of a dog in a bucket"
- **Inference Steps**: 25
- **Blank Image Detection**:
  - Standard deviation threshold: 1e-4 (default)
  - Optional constant value check
  - Outlier tolerance support
- **Image Format**: PNG
- **Test Modes**:
  - `single`: Test individual submodules
  - `compile_except`: Test all submodules except specified path(s)

## Compilation Modes

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

### Using Compile Except Mode

1. **Single Path Exclusion**:
   ```bash
   python vae_diagnostic.py --mode compile_except --exclude_path "path.to.submodule"
   ```

2. **Multiple Path Exclusion**:
   ```bash
   python vae_diagnostic.py --mode compile_except --exclude_path "path1,path2,path3"
   ```

3. **Using Bad Paths File**:
   ```bash
   python vae_diagnostic.py --mode compile_except --exclude_path "bad_paths.txt"
   ```

### Compile Except Mode Results

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

## Best Practices for Compile Except Mode

1. Start with single path exclusion to identify problematic modules
2. Use the bad paths file to track problematic modules
3. Incrementally add paths to exclusion list
4. Monitor module interactions in the results
5. Use TensorBoard to visualize interaction effects
6. Check error messages for module type compatibility
7. Use the CSV output to analyze module relationships
8. Save images to visually verify compilation effects

## Metrics and Logging

The tool provides comprehensive metrics tracking through multiple channels:

1. **TensorBoard Metrics**:
   - Per-module mean values
   - Per-module standard deviation
   - Hierarchical metric organization by module path
   - Test duration tracking
   - Success/failure status

2. **Weights & Biases Metrics**:
   - Module-specific metrics under path-based namespaces
   - Mean and standard deviation tracking
   - Blank image detection status
   - Real-time metric visualization
   - Test configuration tracking
   - Performance metrics

3. **File-based Logging**:
   - JSON format for detailed analysis
   - CSV format for spreadsheet processing
   - Text file for quick reference of problematic modules
   - Hierarchical submodule listing
   - Test summary with statistics

## Error Handling

The tool implements sophisticated error handling:

1. **Compilation Errors**:
   - Detailed error messages with module type information
   - Original module type preservation for debugging
   - Hierarchical error tracking for multi-path tests
   - Non-wrappable module detection (ModuleList, ModuleDict, Sequential)

2. **Image Generation Errors**:
   - Blank image detection with configurable threshold
   - Zero-mean output detection
   - Per-module error status tracking
   - Batch dimension handling
   - Channel validation (1 or 3 channels)
   - Tensor dimension validation

3. **Module Type Errors**:
   - Automatic detection of non-wrappable modules
   - Type compatibility checking
   - Original type preservation for debugging
   - Module hierarchy validation
   - Path existence verification

4. **Input Validation**:
   - Tensor type checking
   - Empty tensor detection
   - Dimension validation
   - Channel count validation
   - Batch handling
   - PIL Image conversion support

## Command-Line Arguments

The VAE diagnostic tool supports the following command-line arguments:

| Argument | Description |
|----------|-------------|
| `--output OUTPUT` | Directory to save diagnostic results and outputs (default: `'vae_diagnostic_output'`) |
| `--device DEVICE` | Device to run the pipeline on (`'hpu'` or `'cpu'`, default: `'hpu'`) |
| `--filter FILTER` | Type of submodules to test (`'all'`, `'leaf'`, or `'non-leaf'`, default: `'all'`) |
| `--mode MODE` | Compilation mode (`'single'` or `'compile_except'`, default: `'single'`) |
| `--exclude_path PATH` | Path to exclude in `'compile_except'` mode |
| `--test_paths PATHS` | Specific submodule paths to test (file or comma-separated list) |
| `--save_images` | Save generated images to the output directory |
| `--model_name NAME` | Name of the pretrained model to use (default: `'stabilityai/stable-diffusion-xl-base-1.0'`) |
| `--gaudi_config PATH` | Path to Gaudi configuration file (default: `'Habana/stable-diffusion'`) |
| `--use_tensorboard` | Enable TensorBoard logging |
| `--use_wandb` | Enable Weights and Biases logging |
| `--wandb_project NAME` | Weights and Biases project name (default: `'vae_diagnostic'`) |
| `--wandb_run NAME` | Weights and Biases run name (default: `'run_vae_test'`) |
| `--list_submodules` | List VAE decoder submodules hierarchically and save to a file |

## Usage Examples

```bash
# Basic diagnostic run with HPU
python vae_diagnostic.py --device hpu --filter all --mode single

# Save generated images with custom model
python vae_diagnostic.py --device hpu --save_images --model_name "custom/model/path"

# Test all except specific path with Gaudi config
python vae_diagnostic.py --mode compile_except --exclude_path "path.to.submodule" --gaudi_config "path/to/config.json"

# Enable TensorBoard logging
python vae_diagnostic.py --use_tensorboard

# Enable Weights & Biases logging
python vae_diagnostic.py --use_wandb --wandb_project "my_project" --wandb_run "test_run"

# List all submodules
python vae_diagnostic.py --list_submodules

# Test specific paths from a file
python vae_diagnostic.py --test_paths "path/to/paths.txt"

# Test specific paths from a comma-separated list
python vae_diagnostic.py --test_paths "path1,path2,path3"
```

## Testing Specific Paths

The tool provides flexible options for testing specific submodule paths:

1. **Using a File**:
   ```bash
   python vae_diagnostic.py --test_paths "path/to/paths.txt"
   ```

2. **Using Comma-Separated List**:
   ```bash
   python vae_diagnostic.py --test_paths "path1,path2,path3"
   ```

## Path Testing Behavior

When testing specific paths:

1. **File Input**:
   - Each line in the file should contain one path
   - Empty lines are ignored
   - Comments (lines starting with #) are ignored
   - Paths are validated before testing

2. **Comma-Separated List**:
   - Paths are split by commas
   - Whitespace is trimmed
   - Empty entries are ignored
   - Each path is validated individually

## Retesting Workflow

The tool supports a retesting workflow for problematic paths:

1. **Initial Test**:
   ```bash
   python vae_diagnostic.py --mode single
   ```

2. **Save Bad Paths**:
   - Review the results
   - Save problematic paths to a file

3. **Retest Bad Paths**:
   ```bash
   # First run a single mode test to generate bad_paths.txt
   python vae_diagnostic.py --mode single

   # Then run compile_except mode without exclude_path to use bad_paths.txt
   python vae_diagnostic.py --mode compile_except
   ```

## Best Practices for Path Testing

1. Start with a small set of paths
2. Use the file input for large path sets
3. Validate paths before testing
4. Monitor memory usage with large path sets
5. Use the CSV output for analysis
6. Save images for visual verification
7. Use TensorBoard for metric tracking
8. Document problematic paths

## Output Structure

The tool generates the following output structure:

```
vae_diagnostic_output/
├── images/
│   ├── module1.png
│   ├── module2.png
│   └── ...
├── metrics/
│   ├── metrics.json
│   └── metrics.csv
├── logs/
│   ├── tensorboard/
│   └── wandb/
└── submodules/
    └── submodule_list.txt
```

## Interpreting Results

1. **Metrics Analysis**:
   - Check mean values for image quality
   - Monitor standard deviation for stability
   - Look for patterns in problematic modules

2. **Image Analysis**:
   - Compare generated images
   - Look for visual artifacts
   - Check for blank or corrupted outputs

3. **Error Analysis**:
   - Review error messages
   - Check module type compatibility
   - Analyze interaction effects

## Test Summary

The tool provides a comprehensive test summary:

1. **Module Statistics**:
   - Total modules tested
   - Successful compilations
   - Failed compilations
   - Blank image outputs

2. **Performance Metrics**:
   - Average mean values
   - Average standard deviation
   - Compilation success rate

3. **Error Summary**:
   - Common error types
   - Problematic module patterns
   - Interaction effects

## Best Practices

1. **Testing Strategy**:
   - Start with single mode
   - Use compile_except for verification
   - Test incrementally
   - Document results

2. **Resource Management**:
   - Monitor memory usage
   - Use appropriate batch sizes
   - Clean up resources

3. **Documentation**:
   - Keep track of tested paths
   - Document problematic modules
   - Save test configurations

4. **Analysis**:
   - Use visualization tools
   - Track metrics over time
   - Compare different configurations
