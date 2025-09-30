
# Colony Project (End-to-end code)

## Setup
```bash
conda create -n colony python=3.10 -y
conda activate colony
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python scikit-image albumentations matplotlib tqdm shapely scipy pandas jsonschema
pip install pytorch-grad-cam pyyaml
```

## Data
- Put images in `data/*/images` and corresponding binary masks as PNG in `data/*/masks` with the same base filename.
- If you have YOLO `.txt` or JSON annotations, convert to masks using utilities in `src/utils/masks.py` and save PNG masks.

## Train
```bash
python train_run.py
```

## Test
```bash
python test_run.py
```

## Notes
- Model: HATUNet (attention-augmented UNet; you can replace bottleneck with Swin).
- Loss: Dice + BCE.
- Post-processing: morphology + watershed refinement.
- Explainability: Grad-CAM++ helper in `src/vis/gradcam.py`.
