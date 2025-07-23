# test.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import PatchDataset
from models.edsr import EDSR
from sklearn.model_selection import train_test_split
import torchvision.utils as vutils
from tqdm import tqdm

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "outputs/checkpoints/edsr_epoch5.pth"
LR_PATCH_DIR = "patches/LR"
HR_PATCH_DIR = "patches/HR"
SAVE_PREDICTIONS = True
OUTPUT_DIR = "outputs/test_results"
BATCH_SIZE = 1

# === Metrics ===
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# === Load test split ===
all_files = sorted(os.listdir(LR_PATCH_DIR))
_, valtest = train_test_split(all_files, test_size=0.2, random_state=42)
_, test_files = train_test_split(valtest, test_size=0.5, random_state=42)

test_set = PatchDataset(LR_PATCH_DIR, HR_PATCH_DIR, test_files)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# === Load model ===
model = EDSR().to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

criterion = nn.L1Loss()

# === Evaluation ===
total_psnr = 0.0
total_loss = 0.0

if SAVE_PREDICTIONS:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Evaluating {len(test_loader)} patches...")

with torch.no_grad():
    for idx, (lr, hr) in enumerate(tqdm(test_loader)):
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)
        sr = model(lr)

        # Metrics
        loss = criterion(sr, hr).item()
        score = psnr(sr, hr).item()
        total_loss += loss
        total_psnr += score

        # Save SR image
        if SAVE_PREDICTIONS:
            save_path = os.path.join(OUTPUT_DIR, f"sample_{idx+1}.png")
            vutils.save_image(sr, save_path)

# === Final Results ===
avg_psnr = total_psnr / len(test_loader)
avg_loss = total_loss / len(test_loader)

print(f"\nTest Results:")
print(f"   Average PSNR : {avg_psnr:.2f} dB")
print(f"   Average L1 Loss : {avg_loss:.6f}")

# Optional: Save to file
with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
    f.write(f"Average L1 Loss: {avg_loss:.6f}\n")
