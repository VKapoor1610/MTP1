from typing import List, Tuple, Optional
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from omegaconf import OmegaConf

# pip install dists-pytorch
from DISTS_pytorch import DISTS
from ldm.xformers_state import disable_xformers
from model.spaced_sampler import SpacedSampler
from model.ddim_sampler import DDIMSampler
from model.diffeic import DiffEIC
from utils.image import pad
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts

# Function to calculate PSNR between two images
def calculate_psnr(original, reconstructed):
    original = original / 255.0  # Convert to [0, 1] range
    reconstructed = reconstructed / 255.0  # Convert to [0, 1] range
    return psnr(original, reconstructed, data_range=1.0)  # PSNR between 0 and 1 range

@torch.no_grad()
def process(
    model: DiffEIC,
    imgs: List[np.ndarray],
    sampler: str,
    steps: int,
    stream_path: str
) -> Tuple[List[np.ndarray], float]:
    n_samples = len(imgs)
    if sampler == "ddpm":
        sampler = SpacedSampler(model, var_type="fixed_small")
    else:
        sampler = DDIMSampler(model)
    control = torch.tensor(np.stack(imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    
    height, width = control.size(-2), control.size(-1)
    bpp = model.apply_condition_compress(control, stream_path, height, width)
    cond = {
        "c_latent": [model.apply_condition_decompress(stream_path)],
        "c_crossattn": [model.get_learned_conditioning([""] * n_samples)]
    }
    
    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    if isinstance(sampler, SpacedSampler):
        samples = sampler.sample(
            steps, shape, cond,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            cond_fn=None, x_T=x_T
        )
    else:
        sampler: DDIMSampler
        samples, _ = sampler.sample(
            S=steps, batch_size=shape[0], shape=shape[1:],
            conditioning=cond, unconditional_conditioning=None,
            x_T=x_T, eta=0
        )
    
    x_samples = model.decode_first_stage(samples)
    x_samples = ((x_samples + 1) / 2).clamp(0, 1)
    
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    
    preds = [x_samples[i] for i in range(n_samples)]
    
    return preds, bpp


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--ckpt_sd", default='./MTP1/weight/emapruned.ckpt', type=str, help="checkpoint path of stable diffusion")
    parser.add_argument("--ckpt_lc", default='path to checkpoint file of lfgcm and control module', type=str, help="checkpoint path of lfgcm and control module")
    parser.add_argument("--config", default='configs/model/diffeic.yaml', type=str, help="model config path")
    parser.add_argument("--input", type=str, default='path to input images')
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", default=50, type=int)
    parser.add_argument("--output", type=str, default='results/')
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    
    if args.device == "cpu":
        disable_xformers()

    model: DiffEIC = instantiate_from_config(OmegaConf.load(args.config))
    ckpt_sd = torch.load(args.ckpt_sd, map_location="cpu")['state_dict']
    ckpt_lc = torch.load(args.ckpt_lc, map_location="cpu")['state_dict']
    
    ckpt_sd.update(ckpt_lc)
    load_state_dict(model, ckpt_sd, map_location="cpu")['state_dict'] , strict=False)
    
    # update preprocess model
    model.preprocess_model.update(force=True)
    model.freeze()
    model.to(args.device)

    bpps = []
    psnr_values = []  # List to store PSNR values
    dists_values = [] # List of DISTS values 
    
    assert os.path.isdir(args.input)
    
    print(f"sampling {args.steps} steps using {args.sampler} sampler")
    
    for file_path in list_image_files(args.input, follow_links=True):
        img = Image.open(file_path).convert("RGB")
        x = pad(np.array(img), scale=64)
        
        save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
        parent_path, stem, _ = get_file_name_parts(save_path)
        stream_parent_path = os.path.join(parent_path, 'data')
        save_path = os.path.join(parent_path, f"{stem}.png")
        stream_path = os.path.join(stream_parent_path, f"{stem}")

        os.makedirs(parent_path, exist_ok=True)
        os.makedirs(stream_parent_path, exist_ok=True)
        
        preds, bpp = process(
            model, [x], steps=args.steps, sampler=args.sampler,
            stream_path=stream_path
        )
        pred = preds[0]

        bpps.append(bpp)
        
        # remove padding
        pred = pred[:img.height, :img.width, :]

        Image.fromarray(pred).save(save_path)
        print(f"save to {save_path}, bpp {bpp}")
        
        # Calculate PSNR for the current image
        psnr_value = calculate_psnr(np.array(img), pred)  # Compare original image with predicted image
        psnr_values.append(psnr_value)
        print(f"PSNR for {file_path}: {psnr_value:.2f} dB")
        
        dists_obj = DISTS() 
        dists_value = dists_obj(img, Image.fromarray(pred))
        dists_values.append(dists_value)
        

    avg_bpp = sum(bpps) / len(bpps)
    avg_psnr = np.mean(psnr_values)  # Calculate average PSNR
    avg_dists = np.mean(dists_values)

    print(f'avg bpp: {avg_bpp}')
    print(f'avg PSNR: {avg_psnr:.2f} dB')
    print(f'avg DISTS: {avg_dists:.2f}')

if __name__ == "__main__":
    main()
