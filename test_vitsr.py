import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision.transforms.functional import resize
from data.dataset import PatchDataset
from models.vit_sr import ViTSR

LR_PATCH_DIR = "patches/LR"
HR_PATCH_DIR = "patches/HR"
CHECKPOINT_PATH = "outputs/checkpoints/vitsr_epoch5.pth"
SAVE_DIR = "outputs/test_results"
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

all_files = sorted(os.listdir(LR_PATCH_DIR))
test_set = PatchDataset(LR_PATCH_DIR, HR_PATCH_DIR, all_files)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model = ViTSR().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

criterion = nn.L1Loss()

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

total_psnr = 0.0
total_l1 = 0.0

print(f"Evaluating {len(test_loader.dataset)} patches...")

for idx, (lr, hr) in enumerate(tqdm(test_loader)):
    lr, hr = lr.to(DEVICE), hr.to(DEVICE)
    with torch.no_grad():
        sr = model(lr)

    # Resize HR to match SR shape for fair metric computation
    hr_resized = resize(hr, size=sr.shape[2:], antialias=True)

    total_psnr += psnr(sr, hr_resized).item()
    total_l1 += criterion(sr, hr_resized).item()

    # Visualization: resize LR to HR for better viewing
    lr_up = resize(lr, size=sr.shape[2:], antialias=True)

    comparison = torch.cat([lr_up.clamp(0, 1), sr.clamp(0, 1), hr_resized.clamp(0, 1)], dim=3)
    save_path = os.path.join(SAVE_DIR, f"sample_{idx+1:04d}.png")
    save_image(comparison, save_path)

avg_psnr = total_psnr / len(test_loader)
avg_l1 = total_l1 / len(test_loader)

print("\nTest Results:")
print(f"  Average PSNR : {avg_psnr:.2f} dB")
print(f"  Average L1 Loss : {avg_l1:.6f}")
print(f"\nComparisons saved to: {SAVE_DIR}")
