import os
import rasterio
import numpy as np
from tqdm import tqdm

PATCH_SIZE_LR = 64
SCALE = 4
PATCH_SIZE_HR = PATCH_SIZE_LR * SCALE
STRIDE = 64  # Can reduce to get more overlap

def extract_patches(lr_path, hr_path, lr_out_dir, hr_out_dir, base_name):
    with rasterio.open(lr_path) as lr_src, rasterio.open(hr_path) as hr_src:
        lr_img = lr_src.read()
        hr_img = hr_src.read()

        h, w = lr_img.shape[1], lr_img.shape[2]

        count = 0
        for i in range(0, h - PATCH_SIZE_LR + 1, STRIDE):
            for j in range(0, w - PATCH_SIZE_LR + 1, STRIDE):
                lr_patch = lr_img[:, i:i+PATCH_SIZE_LR, j:j+PATCH_SIZE_LR]
                hr_patch = hr_img[:, i*SCALE:i*SCALE+PATCH_SIZE_HR, j*SCALE:j*SCALE+PATCH_SIZE_HR]

                # Skip if shapes are invalid (e.g., near borders)
                if lr_patch.shape[1:] != (PATCH_SIZE_LR, PATCH_SIZE_LR):
                    continue
                if hr_patch.shape[1:] != (PATCH_SIZE_HR, PATCH_SIZE_HR):
                    continue

                lr_patch_path = os.path.join(lr_out_dir, f"{base_name}_patch_{count:03d}.tif")
                hr_patch_path = os.path.join(hr_out_dir, f"{base_name}_patch_{count:03d}.tif")

                with rasterio.open(
                    lr_patch_path, 'w',
                    driver='GTiff',
                    height=PATCH_SIZE_LR,
                    width=PATCH_SIZE_LR,
                    count=lr_patch.shape[0],
                    dtype=lr_patch.dtype
                ) as dst:
                    dst.write(lr_patch)

                with rasterio.open(
                    hr_patch_path, 'w',
                    driver='GTiff',
                    height=PATCH_SIZE_HR,
                    width=PATCH_SIZE_HR,
                    count=hr_patch.shape[0],
                    dtype=hr_patch.dtype
                ) as dst:
                    dst.write(hr_patch)

                count += 1

        return count

def run_patch_extraction(lr_folder, hr_folder, lr_out, hr_out):
    os.makedirs(lr_out, exist_ok=True)
    os.makedirs(hr_out, exist_ok=True)

    all_files = sorted(os.listdir(lr_folder))

    total = 0
    for fname in tqdm(all_files):
        lr_path = os.path.join(lr_folder, fname)
        hr_path = os.path.join(hr_folder, fname)
        if not os.path.exists(hr_path):
            print(f"Skipping {fname} — no HR match.")
            continue
        base = os.path.splitext(fname)[0]
        count = extract_patches(lr_path, hr_path, lr_out, hr_out, base)
        total += count

    print(f"\nDone. Total patches: {total}")

if __name__ == "__main__":
    # Input image folders
    LR_FOLDER = 'archive/LR_2m/LR_2m'
    HR_FOLDER = 'archive/HR_0.5m/HR_0.5m'

    # Output patch folders
    LR_OUT = 'patches/LR'
    HR_OUT = 'patches/HR'

    run_patch_extraction(LR_FOLDER, HR_FOLDER, LR_OUT, HR_OUT)
