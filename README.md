# SpND: Spherical-Nested Diffusion Model for Panoramic Image Outpainting

- This is the official repository of the paper "Spherical-Nested Diffusion Model for Panoramic Image Outpainting" in ICML 2025.

## 1. Environment Setup

```sh
conda env create -f env.yaml

conda activate spnd
```

## 2. Data Prepare

### Matterport3D
- Download Matterport3D scans split (skybox image) : https://niessner.github.io/Matterport/.
- Run preprocessing/stitch_mp3d.py to get the Matterport3D dataset
- use 'find /path_to_Matterport3D/*.png > image_Matterport3D.txt' to get image index


### Structured3D
- Download the Structured3D dataset from "https://structured3d-dataset.org/"
- Use image_Structured3D.txt to get the training and testing images
- use 'find /path_to_Structured3D/*.png > image_Structured3D.txt' to get image index

### Prompts
The prompts for Matterport3D and Structured3D datasets are saved in Matter_prompt.json and Structure_prompts.json, respectively.

## 3. Pretrained Weights
- The weights for our SpND model without prompt is in the SpND folder and the weights for the SPND with prompt is in the SPND\_prompt folder.
- All the weights are given in the [Huggingface]().

## 4. Training
### Without Prompts
```sh
CUDA_VISIBLE_DEVICES=0 python train.py --config models/SpND.yaml --data-file image_Matterport3D.txt --gpus 1 --max-epoch 100
```

### With Prompts
```sh
CUDA_VISIBLE_DEVICES=0 python train_prompt.py --config models/SpND.yaml --data-file image_Matterport3D.txt --mask-file masks/center_mask.png --prompt-file Matter_prompt.json --gpus 1 --max-epoch 100
```

### Chaneg the mask
- Modify the down_mask_path config models/SpND.yaml
- Set the --mask-file path for training


## 5. Testing

### Without Prompts
The input mask should match the `down_mask_path` config in the model config file.
```sh
CUDA_VISIBLE_DEVICES=0 python inference_black_new__cfg_diff_mask.py --config models/SpND.yaml --ckpt <path_to_checkpoint> --data-path image_Matterport3D.txt --mask-path masks/center_mask.png --down-mask-path masks/8_down_center_mask.png
```

### With Prompts
The input mask should match the `down_mask_path` config in the model config file.
```sh
python -m torch.distributed.launch --nproc_per_node=1 multi_inference_black_new_cfg_prompt.py --config models/SpND.yaml --ckpt <path_to_checkpoint> --data-path image_Matterport3D.txt --prompt-file Matter_prompt.json --mask-file masks/center_mask.png
```


