import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from data.dataset import PatchDataset
from models.edsr import EDSR
from torch.utils.data import DataLoader
from PIL import Image

# === CONFIG ===
LR_DIR = "patches/LR"
HR_DIR = "patches/HR"
MODEL_PATH = "outputs/checkpoints/edsr_epoch5.pth"  # Change if needed
OUTPUT_DIR = "outputs/visuals"
NUM_SAMPLES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Ensure output directory exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load model ===
model = EDSR().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === Load test set ===
all_files = sorted(os.listdir(LR_DIR))
sample_files = all_files[:NUM_SAMPLES]
dataset = PatchDataset(LR_DIR, HR_DIR, sample_files)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# === Utility to convert tensor to image ===
def tensor_to_img(tensor):
    tensor = tensor.squeeze().cpu().clamp(0, 1)
    return np.transpose(tensor.numpy(), (1, 2, 0))

# === Generate comparisons ===
for i, (lr, hr) in enumerate(loader):
    lr, hr = lr.to(DEVICE), hr.to(DEVICE)

    with torch.no_grad():
        sr = model(lr)

    lr_img = tensor_to_img(lr)
    sr_img = tensor_to_img(sr)
    hr_img = tensor_to_img(hr)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, [lr_img, sr_img, hr_img], ["LR", "SR", "HR"]):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    save_path = os.path.join(OUTPUT_DIR, f"comparison_{i+1:03}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")
