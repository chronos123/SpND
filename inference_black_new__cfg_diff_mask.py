from ast import arg
from share import *

from cldm.model import create_model, load_state_dict
from torch.utils.data import DataLoader
from dataset_panorama import MyDataset
import cv2
from annotator.util import resize_image
import numpy as np
import torch
import einops
#from cldm.ddim_hacked import DDIMSampler
from PIL import Image
from torchvision.transforms import ToPILImage
import os
import datetime
import torch.nn.functional as F
from ldm.models.diffusion.ddim import DDIMSampler
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Subset
from argparse import ArgumentParser
import torchvision


torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.deterministic = False  
print("classifier-free guidance scale = 3.0 for outpainting")


parser = ArgumentParser()
parser.add_argument(
    "--ckpt",
    type=str,
    required=True
)

parser.add_argument(
    "--data-path",
    type=str,
    required=True
)

parser.add_argument(
    "--config",
    type=str,
    required=True,
)

parser.add_argument(
    "--test-set",
    action="store_true"
)

parser.add_argument(
    "--mask-path",
    type=str,
    default="masks/center_mask.png",
)

parser.add_argument(
    "--down-mask-path",
    type=str,
    default="masks/8_down_center_mask.png",
)


args = parser.parse_args()

# Configs
resume_path = args.ckpt
N = 4
Nnum=1
ddim_steps = 50

dataset = MyDataset(args.data_path, args.mask_path)

if args.test_set:
    train_size = int(0.9 * len(dataset))

    val_indices = list(range(train_size, len(dataset)))
    val_subset = Subset(dataset, val_indices)

    dataloader = DataLoader(val_subset, num_workers=1, batch_size=N, shuffle=False)
else:
    dataloader = DataLoader(dataset, num_workers=1, batch_size=N, shuffle=False)

model = create_model(args.config, down_mask_path=args.down_mask_path).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

model_name = os.path.basename(args.config).split(".")[0]

timenow = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
if args.test_set:
    output_dir = os.path.join('./outputs/',f"{timenow}_test")
else:
    output_dir = os.path.join('./outputs/',f"{timenow}_{model_name}")
    
os.makedirs(output_dir, exist_ok=True)
sample_path = output_dir
base_count = len(os.listdir(sample_path))

with torch.no_grad():
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        prompt = batch['txt'][0]

        control =batch["hint"]
        if len(control.shape) == 3:
            control = control[..., None]
        control = control.cuda()
        Nnum, H, W, C = control.shape

        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * N)]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([""] * N)]}
        shape = (4, H//8 , W//8)
        
        mask_img = Image.open(args.down_mask_path).convert("L")
        mask = torchvision.transforms.ToTensor()(mask_img)
        mask = mask.unsqueeze(0).repeat(N, 4, 1, 1)
        mask_img.close()
        mask = mask.to(control.device)
        
        x0, c = model.get_input(batch, "image", bs=N)

        samples, intermediates = ddim_sampler.sample(ddim_steps, N,
                                                     shape, cond, verbose=False, eta=0,
                                                     unconditional_guidance_scale=3.0,
                                                     unconditional_conditioning=un_cond,
                                                     mask=mask,
                                                     x0=x0[:N])
        x_samples = model.decode_first_stage(samples)

        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
        # print(x_samples.shape)

        for i in range(0, N):
            x_sample = x_samples[i]
            x_sample = (255.0 * x_sample).astype(np.uint8)  
            img = Image.fromarray(x_sample)
            img.save(os.path.join(sample_path, prompt.replace(" ", "-")[:-1] +'-' + f"{base_count:05}" + ".png"))
            base_count += 1
            