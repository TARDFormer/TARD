# TARDFormer_auto

Progressive offset accumulation for RGB-D semantic segmentation with deformable ASPP. This repository contains the core model (`TARDFormer_auto`), dataloaders for NYUv2/SUNRGBD, and evaluation/visualization scripts.

## Features
- **Progressive Offset Accumulation**: `ProgressiveOffsetPredictor` accumulates deformable offsets across steps with learned scaling.
- **Deformable ASPP**: Multi-scale fusion with deformable convolutions plus global context.
- **Drop-in Decoder**: `TARDFormer_auto` replaces DeepLabV3+ ASPP with a deformable variant.
- **Ready-to-use Scripts**: NYUv2/SUNRGBD evaluation and single-image visualization.

## Project Structure
```
src/
  TARDFormer_auto.py      # Main model (includes ProgressiveOffsetPredictor, DeformASPP)
  mix_transformer.py      # Backbone components (mit_b0)
  convnext.py             # Backbone components (convnext_tiny)

eval/
  eval.py                 # NYUv2 evaluation
  eval_SUN.py             # SUNRGBD evaluation
  eval_visualizition.py   # Single-image visualization

NYUv2_dataloader.py       # NYUv2 dataset loader
SUNRGBD_dataloader.py     # SUNRGBD dataset loader
requirements.txt          # Minimal dependencies
```

## Requirements
See `requirements.txt` (minimal):
```
torch==2.4.0
torchvision==0.19.0
timm==1.0.15
thop==0.1.1.post2209072238
opencv-python==4.11.0.86
numpy==1.24.4
pillow==10.4.0
matplotlib==3.7.5
```

## Usage

### NYUv2 Evaluation
```
python eval/eval.py \
  --data-dir /path/to/NYUv2 \
  --ckpt /path/to/TARDFormer_auto.pth \
  --model tardformer_auto \
  --split test.txt \
  --visualize
```
- Expects `data-dir` with `images/`, `depths/`, `labels/`, and `test.txt` listing file basenames (no extension).
- Outputs metrics; if `--visualize`, saves colorized predictions to `./result/` by default.

### SUNRGBD Evaluation
```
python eval/eval_SUN.py \
  --data-dir /path/to/SUNRGBD \
  --ckpt /path/to/TARDFormer_auto.pth \
  --model tardformer_auto \
  --split test.txt \
  --visualize
```
- Expects `train_image/`, `train_depth/`, `train_label/`, `test_image/`, `test_depth/`, `test_label/`, and split file (`test.txt`) with basenames.

### Single-Image Visualization
```
python eval/eval_visualizition.py \
  --image /path/rgb.png \
  --depth /path/depth.png \
  --ckpt /path/to/TARDFormer_auto.pth \
  --model tardformer_auto \
  --visualize
```
- Saves `vis_0.png` (RGB | depth colormap | prediction).

## Model Notes
- `TARDFormer_auto` uses:
  - `ProgressiveOffsetPredictor`: predicts mean/logvar/scale for 3×3 deformable offsets (18 channels).
  - `DeformASPP`: progressive deformable convolutions + global pooling, fused and projected.
- Backbones are loaded from `mix_transformer.py` (MiT-b0) and `convnext.py` (ConvNeXt-Tiny). Pretrain loading helper present but commented.

## Dataset Expectations
- NYUv2: `images/*.png`, `depths/*.png`, `labels/*.png`, and split txt with basenames.
- SUNRGBD: `train_image`, `train_depth`, `train_label`, `test_*` counterparts, and split txt with basenames.
- Dataloaders normalize RGB to ImageNet stats; depth normalized (NYUv2: /1000; SUNRGBD: /1000).

## License
This project is licensed under the Apache 2.0 License

## Citation

Part of the code implementation was adapted from ACNet's repository & AsymFormer.

If you use this code, please cite:
```
@inproceedings{song2025tard,
  title={TARD: An Efficient Adaptive Decoder Mechanism with Progressive Offset Accumulation and Cascaded Adaptive Receptive Field Expansion},
  author={Song, J. and Liu, Y. and Xu, J. and Morooka, K. I.},
  booktitle={International Conference on Neural Information Processing},
  pages={464--479},
  address={Singapore},
  publisher={Springer Nature Singapore},
  year={2025},
  month={November}
}
```

