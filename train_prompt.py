from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dataset_panorama import MyDatasetPrompt
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import datetime,os
import torch
from torch.utils.data import Subset
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    default="models/cldm_pano_new_v2.yaml",
)

parser.add_argument(
    "--gpus",
    type=int,
    required=True,
)

parser.add_argument(
    "--max-epoch",
    type=int,
    required=True,
)

parser.add_argument(
    "--ckpt",
    type=str,
    default=None
)

parser.add_argument(
    "--prompt-file",
    type=str,
    required=True,
)

parser.add_argument(
    "--data-file",
    type=str,
    required=True,
)

parser.add_argument(
    "--mask-file",
    type=str,
    required=True,
)

args = parser.parse_args()
config_path = args.config

version = os.path.basename(config_path)
version = version.split(".")[0][9:]

batch_size = 16
logger_freq = 5000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

def get_checkpoint_dir(trainer):
    log_dir = trainer.log_dir  
    checkpoint_dir = os.path.join(log_dir, 'checkpoint')  
    os.makedirs(checkpoint_dir, exist_ok=True) 
    return checkpoint_dir

timenow = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
dir_path = f'./lightning_logs/{timenow}{version}/'
checkpoint_callback = ModelCheckpoint(
    monitor='val/loss_simple_ema',  
    mode='min',  
    save_top_k=3,  
    verbose=True,  
    dirpath=dir_path,  
    filename='{epoch:02d}-{val/loss_simple_ema:.4f}',
    save_last=True  
)


model = create_model(config_path).cpu()

if args.ckpt is not None:
    pass
else:
    resume_path = './models/control_sd15_ini.ckpt'
    model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


dataname=os.path.join("panorama",timenow + f"{version}")
path=os.path.join("./image_log/train",dataname)
os.makedirs(path, exist_ok=True)


# Misc
dataset = MyDatasetPrompt(data_file=args.data_file, mask_file=args.mask_file, prompt_file=args.prompt_file)
# MatterPort default

train_size = int(0.9 * len(dataset))


train_indices = list(range(0, train_size))
val_indices = list(range(train_size, len(dataset)))

train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset, val_indices)

train_dataloader = DataLoader(train_subset, num_workers=8, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_subset, num_workers=8, batch_size=batch_size, shuffle=True)

logger = ImageLogger(batch_frequency=logger_freq,dataname=dataname, increase_log_steps=False)

torch.cuda.empty_cache()

trainer = pl.Trainer(
    strategy="fsdp",
    accelerator="gpu", 
    gpus=args.gpus, 
    precision=16, 
    callbacks=[checkpoint_callback,logger],
    max_epochs=args.max_epoch,
    )

trainer.fit(
    model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=val_dataloader, 
    ckpt_path=args.ckpt,
    )
