# Pipeline Diagnostic Tool

A Python utility for running diagnostics on the submodules of a Stable Diffusion pipeline. This tool helps identify which submodules can be safely compiled using various backends (HPU, CUDA, CPU) without causing issues in image generation.

## Features

- **Submodule Diagnostics**: Test the effect of compiling individual submodules or all submodules except a specified path
- **Image Generation**: Generate images using a diffusion pipeline and check for blank outputs or errors
- **Logging**: Log results to TensorBoard or Weights and Biases for better visualization
- **Submodule Listing**: List all submodules of the pipeline hierarchically
- **State Management**: Automatically manages and restores module states between tests
- **Memory Management**: Efficient memory handling with automatic cleanup between tests
- **Flexible Testing**: Support for testing specific paths or groups of paths
- **Comprehensive Error Handling**: Detailed error reporting and state preservation

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

The pipeline diagnostic tool can be used in several ways:

### List All Submodules

To see all available submodules in the pipeline:

```bash
python pipeline_diagnostic.py --list_submodules
```

This will print the submodule hierarchy and save it to a file in the output directory.

### Test Individual Submodules

To test compilation of individual submodules:

```bash
python pipeline_diagnostic.py --mode single --filter_type all
```

Options for `--filter_type`:
- `all`: Test all submodules
- `leaf`: Test only leaf modules (those with no children)
- `non-leaf`: Test only non-leaf modules (those with children)

### Test Specific Paths

To test specific submodule paths:

```bash
# Test a single path
python pipeline_diagnostic.py --test_paths "path.to.module"

# Test multiple paths (comma-separated)
python pipeline_diagnostic.py --test_paths "path1,path2,path3"

# Test paths from a file (one path per line)
python pipeline_diagnostic.py --test_paths path/to/paths.txt
```

### Test Specific Root Modules

To test specific parts of the pipeline by specifying root modules:

```bash
# Test specific major components
python pipeline_diagnostic.py --root_modules "unet,text_encoder,vae"

# Test specific submodules
python pipeline_diagnostic.py --root_modules "unet.attention,text_encoder.layers"

# Test multiple non-overlapping parts
python pipeline_diagnostic.py --root_modules "unet.down_blocks,unet.up_blocks"
```

The root_modules option allows you to:
- Focus diagnostics on specific parts of the pipeline
- Test multiple components simultaneously
- Combine with other options like filter_type and mode

### Compile All Except

To compile all submodules except specified ones:

```bash
python pipeline_diagnostic.py --mode compile_except --exclude_path "path1,path2"
```

Or provide a file containing paths to exclude:

```bash
python pipeline_diagnostic.py --mode compile_except --exclude_path path/to/exclude_paths.txt
```

### Additional Options

- `--output_dir`: Directory to save diagnostic results (default: "pipeline_diagnostic_output")
- `--device`: Device to run the pipeline on ("hpu", "cuda", or "cpu", default: "hpu")
- `--log_dir`: Directory for log files
- `--tensorboard_dir`: Directory for TensorBoard logs
- `--use_tensorboard`: Enable TensorBoard logging
- `--use_wandb`: Enable Weights and Biases logging
- `--wandb_project`: Weights and Biases project name
- `--model_name`: Name of the pretrained model to use (default: "stabilityai/stable-diffusion-xl-base-1.0")
- `--gaudi_config`: Path to Gaudi configuration file (default: "Habana/stable-diffusion")
- `--bad_paths_file`: File containing known problematic paths to exclude
- `--test_paths`: Specific path(s) to test (comma-separated list or file)
- `--root_modules`: Comma-separated list of module paths to use as roots for diagnostics

## Test Parameters

The diagnostic tool uses the following default parameters for testing:

- **Test Prompt**: "A picture of a dog in a bucket"
- **Inference Steps**: 5
- **Blank Image Detection**:
  - Standard deviation threshold: 0.05
  - Automatic conversion between tensor and PIL Image formats
  - Support for both single images and batches
- **State Management**:
  - Deep copy of module states for clean testing
  - Automatic state restoration between tests
  - Memory cleanup after each test

## Output

The tool generates several output files in the specified output directory:

- `results.json`: Detailed results in JSON format containing:
  - Module path
  - Module type
  - Original module type
  - Test status (passed/failed)
  - Error message (if any)
- `results.csv`: Results in CSV format for easy analysis
- `bad_paths.txt`: List of problematic submodule paths that failed testing
- `pipeline_submodules_list.txt`: Hierarchical list of all submodules
- `images/`: Directory containing generated test images with status in filename:
  - `{module_path}_OK.png`: Successfully generated images
  - `{module_path}_BLANK.png`: Blank or failed images
- `tensorboard/`: TensorBoard logs (if `--use_tensorboard` is used)
- `logs/`: Directory containing detailed test logs

## Example

Here's a complete example that demonstrates various features:

```bash
python pipeline_diagnostic.py \
    --list_submodules \
    --mode single \
    --filter_type leaf \
    --use_tensorboard \
    --use_wandb \
    --wandb_project "pipeline-diagnostics" \
    --output_dir pipeline_test_results \
    --device hpu \
    --model_name "stabilityai/stable-diffusion-xl-base-1.0" \
    --gaudi_config "Habana/stable-diffusion" \
    --bad_paths_file known_bad_paths.txt \
    --test_paths "path/to/test_paths.txt" \
    --log_dir logs \
    --tensorboard_dir tensorboard_logs
```

This example:
1. Lists all submodules in the pipeline
2. Tests compilation of leaf modules only
3. Uses both TensorBoard and Weights & Biases for logging
4. Excludes known problematic paths
5. Tests specific paths from a file
6. Saves results in a custom output directory
7. Uses HPU device with the specified model and configuration
8. Generates detailed logs in separate directories

## Error Handling

The tool implements comprehensive error handling:

1. **Module State Management**:
   - Deep copy of module states for clean testing
   - Automatic state restoration between tests
   - Memory cleanup after each test

2. **Image Generation**:
   - Automatic format conversion (PIL Image ↔ Tensor)
   - Batch dimension handling
   - Blank image detection
   - Error status tracking

3. **Path Testing**:
   - Support for both file and comma-separated path lists
   - Path existence validation
   - Automatic path filtering

4. **Memory Management**:
   - Automatic CUDA cache clearing
   - Pipeline cleanup between tests
   - Resource deallocation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.