from ast import arg
from share import *
from cldm.model import create_model, load_state_dict
from torch.utils.data import DataLoader
from dataset_panorama import MyDatasetPrompt
import cv2
from annotator.util import resize_image
import numpy as np
import torch
import einops
from PIL import Image
from torchvision.transforms import ToPILImage
import os
import datetime
import torch.nn.functional as F
from ldm.models.diffusion.ddim import DDIMSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import Process
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import Subset


def setup_distributed(rank, world_size):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    torch.distributed.destroy_process_group()

def main_worker(rank, world_size, args):
    setup_distributed(rank, world_size)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Dataset and DataLoader
    dataset = MyDatasetPrompt(args.data_path, prompt_file=args.prompt_file, mask_file=args.mask_file)
    stop_point = -1
    # -1 for start
    total_size = len(dataset)
    remain_indicies = list(range(stop_point + 1, total_size))
    dataset = Subset(dataset, remain_indicies)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=8)

    # Model setup
    model = create_model(args.config).to(rank)
    model.load_state_dict(load_state_dict(args.ckpt, location="cuda:{}".format(rank)))
    model = DDP(model, device_ids=[rank], output_device=rank)
    ddim_sampler = DDIMSampler(model.module)

    # Output directory
    timenow = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join('./outputs/', f"{timenow}_{os.path.basename(args.config).split('.')[0]}")
    os.makedirs(output_dir, exist_ok=True)

    # Sampling loop
    base_count = stop_point + 1  # Ensure unique names for each rank
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            prompt = batch['txt'][0]
            control = batch["hint"].cuda(rank)
            N, H, W, C = control.shape
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            cond = {"c_concat": [control], "c_crossattn": [model.module.get_learned_conditioning([prompt] * N)]}
            un_cond = {"c_concat": [control], "c_crossattn": [model.module.get_learned_conditioning([""] * N)]}
            shape = (4, H // 8, W // 8)

            mask = torch.ones((H // 8, W // 8), device=rank)
            mask[1 * H // 32:3 * H // 32, 3 * W // 64:5 * W // 64] = 0
            mask = 1. - mask.unsqueeze(0).unsqueeze(0).repeat(N, 1, 1, 1)

            x0, _ = model.module.get_input(batch, "image", bs=N)
            samples, _ = ddim_sampler.sample(args.ddim_steps, N, shape, cond, verbose=False, eta=0,
                                              unconditional_guidance_scale=3.0,
                                              unconditional_conditioning=un_cond,
                                              mask=mask, x0=x0[:N])

            x_samples = model.module.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0).cpu().permute(0, 2, 3, 1).numpy()

            for i in range(N):
                x_sample = (255.0 * x_samples[i]).astype(np.uint8)
                img = Image.fromarray(x_sample)
                img.save(os.path.join(output_dir, f"outpainting-{base_count+rank:05}.png"))
                base_count += world_size

    cleanup_distributed()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--mask-file", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--prompt-file", type=str, required=True)
    args = parser.parse_args()

    world_size = args.world_size
    torch.multiprocessing.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
