# gaudi_diffusion_lora.py
import torch
import argparse
from pathlib import Path
import numpy as np
from diffusers import LCMScheduler, DPMSolverMultistepScheduler
from optimum.habana.diffusers import (
    GaudiStableDiffusionPipeline,
    GaudiStableDiffusionXLPipeline,
    GaudiEulerDiscreteScheduler,
    GaudiFlowMatchEulerDiscreteScheduler,
    GaudiFluxPipeline,
)
from torchvision.utils import save_image
import logging
import re
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_pipeline(model_id, scheduler="euler", use_habana=True, use_hpu_graphs=True):
    """Initialize Gaudi-optimized pipeline with automatic model detection"""
    try:
        is_flux = "flux" in model_id.lower()
        
        if is_flux:
            logger.info(f"Initializing Flux pipeline for model: {model_id}")
            pipe = GaudiFluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                use_habana=use_habana,
                use_hpu_graphs=use_hpu_graphs,
                gaudi_config="Habana/stable-diffusion",
                bf16_full_eval=True,
            )
            pipe.scheduler = GaudiFlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            return pipe, True
        
        if "xl" in model_id.lower():
            logger.info(f"Initializing SDXL pipeline for model: {model_id}")
            pipe = GaudiStableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                use_habana=use_habana,
                use_hpu_graphs=use_hpu_graphs,
                gaudi_config="Habana/stable-diffusion",
                bf16_full_eval=True,
            )
            pipe.scheduler = GaudiEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            return pipe, False
        
        # Standard SD configuration
        logger.info(f"Initializing standard SD pipeline for model: {model_id}")
        pipe = GaudiStableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            use_habana=use_habana,
            use_hpu_graphs=use_hpu_graphs,
            gaudi_config="Habana/stable-diffusion",
            bf16_full_eval=True,
        )
        
        if scheduler == "dpm":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "lcm":
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        
        return pipe, False
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        raise


def generate_comparison_grid(images, titles, output_file):
    """Create side-by-side comparison of generated images with titles"""
    try:
        # Convert PIL images to tensors and ensure proper dimensions
        tensors = []
        for img in images:
            # Convert PIL image to tensor and normalize to [0, 1]
            if not isinstance(img, torch.Tensor):
                img_tensor = torch.tensor(np.array(img)).float()
                if img_tensor.dim() == 3:  # If image is HWC
                    img_tensor = img_tensor.permute(2, 0, 1)  # Convert to CHW
                img_tensor = img_tensor / 255.0
            else:
                img_tensor = img
                if img_tensor.dim() == 3 and img_tensor.shape[0] != 3:  # If tensor is HWC
                    img_tensor = img_tensor.permute(2, 0, 1)  # Convert to CHW
                if img_tensor.max() > 1.0:  # Normalize if not already normalized
                    img_tensor = img_tensor / 255.0
            
            # Ensure tensor is in correct format (C, H, W) and normalized
            if img_tensor.dim() != 3 or img_tensor.shape[0] != 3:
                raise ValueError(f"Invalid image tensor shape: {img_tensor.shape}")
            
            tensors.append(img_tensor)
        
        # Stack tensors horizontally
        grid = torch.cat(tensors, dim=2)
        
        # Convert grid tensor to PIL image for drawing titles
        grid_pil = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        draw = ImageDraw.Draw(grid_pil)
        
        # Load a default font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default(20)
        
        # Draw titles on each part of the grid
        width_per_image = grid_pil.width // len(images)
        for i, title in enumerate(titles):
            x = i * width_per_image + 10
            y = 10
            draw.text((x, y), title, fill="white", font=font)
        
        # Save the grid
        #save_image(grid, output_file)
        grid_pil.save(output_file)
        logger.info(f"Saved comparison grid with titles to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate comparison grid: {str(e)}")
        raise 


def main():
    parser = argparse.ArgumentParser(description="Gaudi-optimized SD with LoRA support")
    parser.add_argument("--model_id", required=True, 
                       help="Model ID (e.g. stabilityai/stable-diffusion-xl-base-1.0)")
    parser.add_argument("--lora_paths", nargs='+', default=[], 
                       help="Paths to LoRA adapters")
    parser.add_argument("--scheduler", choices=["euler", "dpm", "lcm"], default="euler",
                       help="Sampling scheduler")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Text prompt for generation")
    parser.add_argument("--output_dir", default="outputs",
                       help="Output directory for images")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    args = parser.parse_args()

    try:
        # Set random seed if provided
        if args.seed is not None:
            torch.manual_seed(args.seed)
            logger.info(f"Set random seed to {args.seed}")

        # Prompt input handling
        prompt = args.prompt
        if not prompt:
            prompt = input("Enter your text prompt for image generation: ").strip()
            if not prompt:
                raise ValueError("Error: Prompt cannot be empty.")

        # Initialize pipeline
        pipe, is_flux = initialize_pipeline(
            args.model_id,
            scheduler=args.scheduler,
            use_habana=True,
            use_hpu_graphs=True
        )
        pipe.to("hpu")

        # Load LoRA adapters
        for lora_path in args.lora_paths:
            try:
                pipe.load_lora_weights(lora_path)
                logger.info(f"Loaded LoRA: {Path(lora_path).name}")
            except Exception as e:
                logger.error(f"Failed to load LoRA {lora_path}: {str(e)}")
                raise

        # Generation parameters
        base_params = {
            "prompt": prompt,
            "num_inference_steps": 30 if args.scheduler == "lcm" else 50,
            "height": 1024, # 768 if is_flux else 512,
            "width": 1024, #1360 if is_flux else 512,
            "num_images_per_prompt": 1,
        }

        # Generate images
        images = []
        try:
            if args.lora_paths:
                # If LoRA loaded, disable/unfuse for capture base model
                pipe.disable_lora()  # Enable the LoRA
                pipe.unfuse_lora()   # Fuse the LoRA into the model
            base_image = pipe(**base_params).images[0]
            images.append(base_image)
            titles = ["Base Model"]

            if args.lora_paths:
                # Apply the LoRA (Enable/Fuse)
                pipe.enable_lora()  # Enable the LoRA
                pipe.fuse_lora()     # Fuse the LoRA into the model
                # LoRA-modified generation
                lora_image = pipe(**base_params, cross_attention_kwargs={"scale": 0.8}).images[0]
                images.append(lora_image)
                titles.append("LoRA Enhanced")
                
                # Apply the LoRA
                #pipe.disable_lora()  # Enable the LoRA
                pipe.unfuse_lora()   # Fuse the LoRA into the model
                # Restore weights and verify
                pipe.unload_lora_weights()
                restored_image = pipe(**base_params).images[0]
                images.append(restored_image)
                titles.append("Weights Restored")

            # Create a safe subfolder nae from the model_id
            model_name_safe = re.sub(r'[^a-zA-Z0-9._-]', '_', args.model_id.split('/')[-1])
            output_dir = Path(args.output_dir) / model_name_safe
            output_dir.mkdir(parents=True, exist_ok=True)
            generate_comparison_grid(images, titles, output_dir / "comparison_grid.png")
            logger.info(f"✅ Results saved to {output_dir}/")
            
        except Exception as e:
            logger.error(f"Failed during image generation: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
