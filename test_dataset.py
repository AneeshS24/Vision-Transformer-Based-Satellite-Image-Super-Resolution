from data.dataset import GeoTIFFSuperResolutionDataset
from torch.utils.data import DataLoader

# Updated path
lr_root = 'archive/LR_2m/LR_2m'
hr_root = 'archive/HR_0.5m/HR_0.5m'

dataset = GeoTIFFSuperResolutionDataset(lr_root, hr_root)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

print(f"\n🔢 Total paired samples: {len(dataset)}\n")

for lr, hr in loader:
    print("📦 LR shape:", lr.shape)
    print("🏁 HR shape:", hr.shape)
    break
