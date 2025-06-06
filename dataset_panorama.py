import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_file="image_Matterport3D.txt", mask_file="masks/center_mask.png"):
        with open(data_file, "r") as file:
            self.image_paths = file.read().splitlines()
        self.mask = cv2.imread(mask_file)
         
    def __len__(self):
        return len(self.image_paths)
    
    def crop_image_with_mask(self, image):
        mask = self.mask.astype(np.bool)

        cropped_image = np.zeros_like(image)
        cropped_image[mask] = image[mask]

        return cropped_image

    def __getitem__(self, idx):
        prompt="outpainting"
        target = cv2.imread(self.image_paths[idx])
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        source = self.crop_image_with_mask(target)
        
        target = (target.astype(np.float32) / 127.5) - 1.0
        
        source = source.astype(np.float32) / 255.0
        return dict(jpg=target, txt=prompt, hint=source)


class MyDatasetPrompt(Dataset):
    def __init__(self, data_file="image_Matterport3D.txt", mask_file="masks/center_mask.png", prompt_file="Matter_prompt.json"):
        with open(data_file, "r") as file:
            self.image_paths = file.read().splitlines()
        self.mask = cv2.imread(mask_file)
        with open(prompt_file, "r") as f:
            self.pair_data = json.load(f)
         
    def __len__(self):
        return len(self.image_paths)
    
    def crop_image_with_mask(self, image):
        mask = self.mask.astype(np.bool)

        cropped_image = np.zeros_like(image)
        cropped_image[mask] = image[mask]

        return cropped_image

    def __getitem__(self, idx):
        name = os.path.basename(self.image_paths[idx])
        prompt = self.pair_data[name]
        target = cv2.imread(self.image_paths[idx])
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        source = self.crop_image_with_mask(target)
        
        target = (target.astype(np.float32) / 127.5) - 1.0

        source = source.astype(np.float32) / 255.0
        return dict(jpg=target, txt=prompt, hint=source)


if __name__ == "__main__":
    dataset = MyDataset()
    for e in dataset:
        pass
    pass