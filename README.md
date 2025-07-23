# Vision-Transformer-Based-Satellite-Image-Super-Resolution
A deep learning pipeline for 4× super-resolution of satellite imagery using both EDSR and Vision Transformer (ViT) based architectures. Trained on GeoTIFF satellite images with full automation for training, testing, and visualization.
Project Structure
superresolution/
├── archive/              # Raw HR (0.5m) and LR (2m) GeoTIFF satellite images
├── data/                 # Dataset and patch preprocessing code
│   ├── dataset.py
│   └── preprocess.py
├── models/               # Model architectures
│   ├── edsr.py           # EDSR: Enhanced Deep Residual Network
│   └── vit_sr.py         # Vision Transformer for Super-Resolution
├── outputs/              # Generated results
│   ├── checkpoints/      # Trained model weights
│   ├── test_results/     # Inference outputs
│   └── visuals/          # Side-by-side comparison images
├── patches/              # Preprocessed LR/HR image patches for training
│   ├── HR/
│   └── LR/
├── config.yaml           # Global configuration
├── train.py              # Train both EDSR or ViTSR
├── test_edsr.py          # Inference with EDSR model
├── test_vitsr.py         # Inference with ViTSR model
├── evaluate.py           # Compute PSNR on validation/test sets
├── visualize.py          # Generate visual comparison grids
├── run_all.py            # Complete automated pipeline runner
└── test_dataset.py       # Sanity check for patch datasets

Model	Description
EDSR	Enhanced Deep Residual Network (baseline CNN model for SR)
ViTSR	Custom Vision Transformer-based architecture for 4× upscaling

Results
Metric           	EDSR     	ViTSR (Transformer)
Validation PSNR	~25.2 dB	25.95 dB

