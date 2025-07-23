import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data.dataset import PatchDataset
from models.vit_sr import ViTSR

LR_PATCH_DIR = "patches/LR"
HR_PATCH_DIR = "patches/HR"
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_files = sorted(os.listdir(LR_PATCH_DIR))
train_files, valtest_files = train_test_split(all_files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(valtest_files, test_size=0.5, random_state=42)

print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

train_set = PatchDataset(LR_PATCH_DIR, HR_PATCH_DIR, train_files)
val_set = PatchDataset(LR_PATCH_DIR, HR_PATCH_DIR, val_files)
test_set = PatchDataset(LR_PATCH_DIR, HR_PATCH_DIR, test_files)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1)
test_loader = DataLoader(test_set, batch_size=1)

model = ViTSR().to(DEVICE)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for lr, hr in train_loader:
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)
        sr = model(lr)
        if sr.shape != hr.shape:
            sr = nn.functional.interpolate(sr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        loss = criterion(sr, hr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    val_psnr = 0.0
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            sr = model(lr)
            if sr.shape != hr.shape:
                sr = nn.functional.interpolate(sr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
            val_psnr += psnr(sr, hr).item()

    avg_psnr = val_psnr / len(val_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_loss:.4f} | Val PSNR: {avg_psnr:.2f} dB")

    ckpt_dir = os.path.join("outputs", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"vitsr_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to: {ckpt_path}")
