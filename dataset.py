import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class PatchDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, file_list, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.file_list = file_list
        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        lr_path = os.path.join(self.lr_dir, fname)
        hr_path = os.path.join(self.hr_dir, fname)

        # Load as PIL (or use rasterio if needed)
        lr_img = Image.open(lr_path)
        hr_img = Image.open(hr_path)

        # Convert to tensor [C, H, W]
        lr = self.transform(lr_img)
        hr = self.transform(hr_img)

        return lr, hr
