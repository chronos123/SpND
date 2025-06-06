## Data Prepare

### Matterport3D
- Download Matterport3D scans split (skybox image) : https://niessner.github.io/Matterport/.
- Run preprocessing/stitch_mp3d.py to get the Matterport3D dataset

### Structured3D
- Download the Structured3D dataset from "https://structured3d-dataset.org/"
- Use image_Structured3D.txt to get the training and testing images

### Prompts
The prompts for Matterport3D and Structured3D datasets are saved in Matter_prompt.json and Structure_prompts.json, respectively.

## Pretrained Weights
Comming soon!

## Training
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


## Testing
Comming soon!


