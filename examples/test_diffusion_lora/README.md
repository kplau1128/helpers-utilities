# Gaudi Diffusion LoRA Comparison

This repository provides a flexible, high-performance script for running text-to-image generation using Stable Diffusion (SD), Stable Diffusion XL (SDXL), and FLUX models on Habana Gaudi hardware. The script supports LoRA adapters for style transfer and enables visual comparison between base, LoRA-modified, and restored model outputs. It is optimized for Gaudi acceleration and supports various schedulers.

---

## Features

- **Supports SD 1.x, SD 2.x, SDXL, and FLUX models**
- **Gaudi-optimized pipelines** for maximum performance on Habana hardware
- **LoRA adapter support** for easy style transfer and fine-tuning
- **Multiple schedulers**: Euler, DPM, LCM
- **Side-by-side image comparison**: base, LoRA, and restored weights
- **Interactive or CLI prompt entry**
- **Automatic model type detection and configuration**
- **Batch processing support** for efficient generation of multiple images
- **Model offloading** for handling large images and complex prompts
- **Titles for each part of the comparison grid**

---

## Requirements

- Habana Gaudi hardware (Gaudi2 recommended)
- [Habana SynapseAI™ software stack](https://developer.habana.ai/software/)
- Python 3.8 or higher
- CUDA-compatible GPU (for local development and testing)
- Sufficient disk space for model weights and generated images

---

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/habana/gaudi-diffusion-lora.git
    cd gaudi-diffusion-lora
    ```

2. **Create and activate a virtual environment** (recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**
    > **Note:** You must have access to a system with Habana Gaudi hardware and the [Habana SynapseAI™ software stack](https://developer.habana.ai/software/) installed.

    ```bash
    pip install optimum-habana diffusers[torch] accelerate torchvision
    ```

4. **Verify installation**
    ```bash
    python -c "from optimum.habana import GaudiConfig; print('Habana SynapseAI is properly installed')"
    ```

---

## Usage

### Basic Example

```bash
python diffusion_lora.py \
--model_id stabilityai/stable-diffusion-2-1 \
--prompt "A futuristic city skyline at sunset"
```

### With SDXL and LoRA

```bash
python diffusion_lora.py \
--model_id stabilityai/stable-diffusion-xl-base-1.0 \
--lora_paths path/to/lora_adapter \
--prompt "A steampunk airship in a stormy sky" \
--scheduler euler
```

### With FLUX Model

```bash
python diffusion_lora.py \
--model_id black-forest-labs/FLUX.1-dev \
--prompt "A watercolor landscape of a mountain lake" \
--scheduler lcm
```

### Interactive Prompt

If you omit `--prompt`, the script will ask for it interactively:

```bash
python diffusion_lora.py --model_id stabilityai/stable-diffusion-2-1
Enter your prompt when asked
```

### Batch Processing

```bash
python diffusion_lora.py \
--model_id stabilityai/stable-diffusion-2-1 \
--prompt "A futuristic city skyline at sunset" \
--num_images 4 \
--batch_size 2
```

---

## Output

- The script saves a side-by-side comparison grid of generated images in the specified `--output_dir` (default: `outputs/`).
- The grid includes:
    - **Base Model Output**
    - **LoRA-Enhanced Output** (if LoRA used)
    - **Weights Restored Output** (after unloading LoRA)
- Each image is saved with a timestamp and prompt hash for easy reference
- A JSON metadata file is generated alongside the images containing generation parameters
- Titles are added to each part of the comparison grid for clarity

---

## Arguments

| Argument         | Description                                                        | Default     |
|------------------|--------------------------------------------------------------------|-------------|
| `--model_id`     | Model identifier (e.g., SD/SDXL/FLUX Hugging Face repo or path)    | Required    |
| `--lora_paths`   | Paths to one or more LoRA adapters (optional)                      | None        |
| `--scheduler`    | Sampling scheduler: `euler`, `dpm`, or `lcm`                       | `euler`     |
| `--prompt`       | Text prompt for image generation (optional, interactive if omitted)| None        |
| `--output_dir`   | Output directory for images                                        | `outputs/`  |
| `--num_images`   | Number of images to generate                                       | 1           |
| `--batch_size`   | Batch size for processing                                          | 1           |
| `--seed`         | Random seed for reproducibility                                    | Random      |
| `--steps`        | Number of inference steps                                          | 50          |

---

## Example Output

> **Note:** The example comparison grid image will be generated when you run the script. The image shown below is a placeholder.

```bash
PT_HPU_LAZY_MODE=1 python ./test_diffusion_lora/diffusion_lora.py \
--model_id "black-forest-labs/FLUX.1-dev" \
--prompt "A picture of a dog in a bucket" \
--lora_paths users_lora_models/dog_lora_flux_1_bf16
```

![Example Comparison Grid](outputs\FLUX.1-dev\comparison_grid.png)

---

## Performance Tips

1. **Batch Processing**
   - Use `--batch_size` to process multiple images simultaneously
   - Optimal batch size depends on your Gaudi hardware configuration

2. **Model Selection**
   - SDXL provides higher quality but requires more memory
   - FLUX models are optimized for speed with LCM scheduler

3. **LoRA Usage**
   - Multiple LoRA adapters can be combined for style blending
   - Use `--lora_paths` with comma-separated paths

4. **Memory Management**
   - Enable model offloading for large images
   - Monitor memory usage with `htop` or similar tools

---

## Troubleshooting

1. **Out of Memory Errors**
   - Reduce batch size
   - Enable model offloading
   - Use a smaller model variant

2. **Slow Generation**
   - Check if using the optimal scheduler for your model
   - Verify Gaudi hardware is properly configured
   - Consider using FLUX models for faster generation

3. **LoRA Loading Issues**
   - Verify LoRA adapter compatibility with the base model
   - Check file permissions and paths
   - Ensure correct model type detection

---

## Notes

- **FLUX Models:** The script auto-detects FLUX models and applies the correct scheduler and resolution.
- **LoRA:** Multiple LoRA adapters can be loaded for blending styles.
- **Performance:** Optimized for Gaudi2 HPUs; batch processing and model offloading are supported for large images.

---

## License



---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

---

## Acknowledgements

- [Habana Labs](https://habana.ai/)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Black Forest Labs FLUX](https://huggingface.co/black-forest-labs)

