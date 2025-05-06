import torch
import torch.nn as nn
import argparse
import json
import csv
import os
import copy
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False

from diffusers import StableDiffusionXLPipeline
from optimum.habana.diffusers import (
    GaudiStableDiffusionXLPipeline,
    GaudiEulerDiscreteScheduler,
)

#
class CompiledWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.compiled = torch.compile(module, backend="hpu_backend")
        self.original_type = type(module)
    
    def forward(self, *args, **kwargs):
        return self.compiled(*args, **kwargs)

def get_all_submodules(module, prefix=""):
    for name, sub in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        yield path, sub
        yield from get_all_submodules(sub, path)

def filter_submodules(modules, filter_type):
    if filter_type == "all":
        return modules
    elif filter_type == "leaf":
        return [(p, m) for p, m in modules if not any(m.children())]
    elif filter_type == "non-leaf":
        return [(p, m) for p, m in modules if any(m.children())]
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

def is_wrappable_module(module):
    # Skip types like ModuleList, ModuleDict, Sequential, etc
    non_wrappable_types = (nn.ModuleList, nn.ModuleDict, nn.Sequential)
    return isinstance(module, nn.Module) and not isinstance(module, non_wrappable_types)

def apply_compile_to_path(module, target_path, prefix=""):
    for name, sub in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if path == target_path and callable(sub) and is_wrappable_module(sub):
            setattr(module, name, CompiledWrapper(sub))
        else:
            apply_compile_to_path(sub, target_path, path)

def apply_compile_except(module, skip_path, prefix=""):
    for name, sub in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if path != skip_path and callable(sub) and is_wrappable_module(sub):
            setattr(module, name, CompiledWrapper(sub))
        else:
            apply_compile_except(sub, skip_path, path)

def get_submodule_type(model, path):
    submod = model
    for part in path.split('.'):
        submod = getattr(submod, part)
    return type(submod)

def get_submodule_orig_type(model, path):
    submod = model
    for part in path.split('.'):
        submod = getattr(submod, part)
    return getattr(submod, 'original_type', type(submod))

def is_blank(image_tensor, threshold=1e-4):
    #return image_tensor.mean().item() < threshold
    return image_tensor.std().item() < threshold

def save_image_tensor(image_tensor, path):
    image_tensor.save(path)


def list_submodules(model, prefix='', depth=0):
    lines = []
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        indent = ' ' * depth
        type_str = f"{type(module).__module__}.{type(module).__name__}"
        lines.append(f"{indent}- {full_name}: {type_str}")
        lines.extend(list_submodules(module, full_name, depth + 1))
    return lines
        
# --- Create Pipeline ---
def create_pipeline(model_name, gaudi_config=None, device="cpu"):
    if device == "hpu":
        print("[LOAD] GaudiStableDiffusionXLPipeline...")
        scheduler = GaudiEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipe = GaudiStableDiffusionXLPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=False,
            gaudi_config=gaudi_config,
            torch_dtype=torch.bfloat16,
        )
        pipe.set_progress_bar_config(disable=True)
    else:
        print("[LOAD] StableDiffusionXLPipeline...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
        ).to(device)

    return pipe.to(device)

# ------------------ Main Diagnostic Runner ------------------

def run_diagnostic(model_name, gaudi_config, device, mode, filter_type, exclude_path,
                   save_images, output_dir, tb_writer, wandb_run):

    pipe = create_pipeline(model_name, gaudi_config, device)
    vae_decoder_orig = pipe.vae.decoder
    all_submodules = list(get_all_submodules(pipe.vae.decoder))
    submodules = filter_submodules(all_submodules, filter_type)
    if save_images:
        os.makedirs(output_dir, exist_ok=True)

    results = []
    bad_paths = []
    prompt = "A picture of a dog in a bucket"
    num_inference_steps=25

    if mode == "compile_except":
        print(f"[MODE] Compiling all except: {exclude_path}")
        pipe.vae.decoder = copy.deepcopy(vae_decoder_orig).to(device)
        try:
            apply_compile_except(pipe.vae.decoder, exclude_path)
            #image = vae.decode(latent_input)
            image = pipe(prompt=prompt, num_inference_steps=num_inference_steps).images[0]
            img_tensor = ToTensor()(image)
            blank = is_blank(img_tensor, 0.05)
            result = {
                "mode": "compile_except",
                "path": exclude_path,
                "blank": blank,
                "mean": img_tensor.mean().item(),
                "std": img_tensor.std().item(),
                "error": ""
            }
            if blank or result["mean"] == 0:
                bad_paths.append(exclude_path)
            if save_images:
                status = "BLANK" if blank else "OK"
                save_image_tensor(image, os.path.join(output_dir, f"except_{exclude_path.replace('.', '_')}_{status}.png"))
            if tb_writer:
                tb_writer.add_scalar("mean", result["mean"], f"except_{exclude_path}")
                tb_writer.add_scalar("std", result["std"], f"except_{exclude_path}")
            if wandb_run:
                wandb_run.log({f"{exclude_path}/mean": result["mean"],
                               f"{exclude_path}/std": result["std"],
                               f"{exclude_path}/blank": blank})
        except Exception as e:
            err_msg = str(e)
            if "is not iterable" in str(e) and "CompiledWrapper" in str(e):
                orig_type = get_submodule_orig_type(pipe.vae.decoder, exclude_path)
                err_msg += f" ({orig_type.__module__}.{orig_type.__name__})"
            result = {
                "mode": "compile_except",
                "path": exclude_path,
                "blank": True,
                "mean": 0,
                "std": 0,
                "error": err_msg
            }
            bad_paths.append(exclude_path)
        results.append(result)
    else:
        for path, _ in tqdm(submodules, desc="Testing submodules"):
            # pipe = create_pipeline(model_name, gaudi_config, device)
            pipe.vae.decoder = copy.deepcopy(vae_decoder_orig).to(device)
            try:
                apply_compile_to_path(pipe.vae.decoder, path)
                #image = vae.decode(latent_input)
                image = pipe(prompt=prompt, num_inference_steps=num_inference_steps).images[0]
                img_tensor = ToTensor()(image)
                blank = is_blank(img_tensor, 0.05)
                result = {
                    "mode": "single",
                    "path": path,
                    "blank": blank,
                    "mean": img_tensor.mean().item(),
                    "std": img_tensor.std().item(),
                    "error": ""
                }
                if blank or result["mean"] == 0:
                    bad_paths.append(path)
                if save_images:
                    status = "BLANK" if blank else "OK"
                    save_image_tensor(image, os.path.join(output_dir, f"{path.replace('.', '_')}_{status}.png"))
                if tb_writer:
                    tb_writer.add_scalar("mean", result["mean"], path)
                    tb_writer.add_scalar("std", result["std"], path)
                if wandb_run:
                    wandb_run.log({f"{path}/mean": result["mean"],
                                   f"{path}/std": result["std"],
                                   f"{path}/blank": blank})
            except Exception as e:
                err_msg = str(e)
                if "is not iterable" in str(e) and "CompiledWrapper" in str(e):
                    orig_type = get_submodule_orig_type(pipe.vae.decoder, path)
                    err_msg += f" ({orig_type.__module__}.{orig_type.__name__})"
                result = {
                    "mode": "single",
                    "path": path,
                    "blank": True,
                    "mean": 0,
                    "std": 0,
                    "error": err_msg
                }
                bad_paths.append(path)
            results.append(result)

    return results, bad_paths


def save_results(results, bad_paths, json_path, csv_path, bad_paths_path):
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    with open(bad_paths_path, "w") as f:
        for path in bad_paths:
            f.write(path + "\n")

# ------------------ Entry ------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="vae_diagnostic_output")
    parser.add_argument("--device", type=str, default="hpu")
    parser.add_argument("--filter", type=str, default="all", choices=["all", "leaf", "non-leaf"])
    parser.add_argument("--mode", type=str, default="single", choices=["single", "compile_except"])
    parser.add_argument("--exclude_path", type=str, help="Path to exclude in compile_except mode")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="vae_diagnostic")
    parser.add_argument("--wandb_run", type=str, default="run_vae_test")
    parser.add_argument("--list-submodules", action="store_true", help="List VAE decoder submodules hierarchically")
    parser.add_argument("--list-submodules-out", type=str, default="vae_decoder_submodules.txt", help="Write submodule list to specified file")
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.output, exist_ok=True)

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output, "tensorboard")) if args.tensorboard else None

    # Weights & Biases
    wandb_run = None
    if args.wandb:
        if WANDB_AVAILABLE:
            wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run)
        else:
            print("[WARN] wandb not installed. Skipping wandb logging.")

    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    gaudi_config = "Habana/stable-diffusion"
    
    if args.list_submodules or args.list_submodules_out:
        pipe = create_pipeline(model_name=model_name, gaudi_config=gaudi_config)
        lines = list_submodules(pipe.vae.decoder)
        
        if args.list_submodules:
            print("VAE Decoder Submodules:")
            for line in lines:
                print(line)
        if args.list_submodules_out:
            with open(args.list_submodules_out, "w") as f:
                f.write("VAE Decoder Submodules:\n")
                f.write("\n".join(lines))
            print(f"Submodule list written to: {args.list_submodules_out}")

    print("[RUN] Diagnostic...")
    results, bad_paths = run_diagnostic(
        model_name=model_name,
        gaudi_config=gaudi_config,
        device=device,
        mode=args.mode,
        filter_type=args.filter,
        exclude_path=args.exclude_path,
        save_images=args.save_images,
        output_dir=os.path.join(args.output, "images") if args.save_images else None,
        tb_writer=tb_writer,
        wandb_run=wandb_run
    )

    if tb_writer:
        tb_writer.flush()
        tb_writer.close()
    if wandb_run:
        wandb_run.finish()

    json_path = os.path.join(args.output, "results.json")
    csv_path = os.path.join(args.output, "results.csv")
    bad_path = os.path.join(args.output, "bad_submodules.txt")
    save_results(results, bad_paths, json_path, csv_path, bad_path)

    print("\n===== Summary =====")
    print(f"Total tested: {len(results)}")
    print(f"Blank outputs / errors: {len(bad_paths)}")
    print(f"Saved bad submodules to: {bad_path}")
    print("===================\n")

if __name__ == "__main__":
    main()
