# Pipeline Diagnostic Tool

A Python utility for running diagnostics on the submodules of a Stable Diffusion pipeline. This tool helps identify which submodules can be safely compiled using various backends (HPU, CUDA, CPU) without causing issues in image generation.

## Features

- **Submodule Diagnostics**: Test the effect of compiling individual submodules or all submodules except a specified path
- **Image Generation**: Generate images using a diffusion pipeline and check for blank outputs or errors
- **Logging**: Log results to TensorBoard or Weights and Biases for better visualization
- **Submodule Listing**: List all submodules of the pipeline hierarchically
- **State Management**: Automatically manages and restores module states between tests
- **Memory Management**: Efficient memory handling with automatic cleanup between tests

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

## Output

The tool generates several output files in the specified output directory:

- `results.json`: Detailed results in JSON format
- `results.csv`: Results in CSV format
- `bad_paths.txt`: List of problematic submodule paths
- `pipeline_submodules_list.txt`: Hierarchical list of all submodules
- `images/`: Directory containing generated test images with status in filename (OK/BLANK)
- `tensorboard/`: TensorBoard logs (if `--use_tensorboard` is used)

## Example

Here's a complete example that:
1. Lists all submodules
2. Tests compilation of leaf modules
3. Uses TensorBoard for logging
4. Excludes known problematic paths

```bash
python pipeline_diagnostic.py \
    --list_submodules \
    --mode single \
    --filter_type leaf \
    --use_tensorboard \
    --output_dir pipeline_test_results \
    --bad_paths_file known_bad_paths.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 