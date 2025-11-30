# Progressive Offset Accumulation Module

This repository contains the core contribution from the RoDD paper: a progressive offset accumulation mechanism for enhanced multimodal feature fusion in depth estimation tasks.

## Core Components

### 1. ProgressiveOffsetPredictor
A progressive offset predictor that generates deformable convolution offsets with uncertainty estimation. It uses a probabilistic approach to predict mean (μ) and variance (σ²) for offset generation, with support for progressive accumulation.

**Key Features:**
- Uncertainty-aware offset prediction
- Progressive offset accumulation across layers
- Bayesian sampling from Gaussian distribution

### 2. DeformASPP
Deformable Atrous Spatial Pyramid Pooling with progressive offset accumulation. It applies multiple deformable convolution layers with accumulated offsets to capture multi-scale contextual information adaptively.

**Key Features:**
- Progressive deformable convolutions
- Multi-scale feature fusion
- Uncertainty estimation at each step

## Installation

```bash
pip install torch torchvision
```

## Usage

```python
import torch
from progressive_offset_accumulation import ProgressiveOffsetPredictor, DeformASPP

# Example usage
batch_size, in_channels, height, width = 2, 256, 32, 32
out_channels = 128

# Create sample input
x = torch.randn(batch_size, in_channels, height, width)

# Test ProgressiveOffsetPredictor
bop = ProgressiveOffsetPredictor(in_channels)
offset, mu, sigma = bop(x)

# Test with progressive accumulation
offset2, mu2, sigma2 = bop(x, prev_offset=offset)

# Test DeformASPP
daspp = DeformASPP(in_channels, out_channels, steps=3)
features, offset_list, mu_list, sigma_list = daspp(x)
```

## Input/Output Specifications

### ProgressiveOffsetPredictor
- **Input:**
  - `x`: Feature map of shape `(B, C, H, W)`
  - `prev_offset`: Optional previous offset for accumulation, shape `(B, 18, H, W)`

- **Output:**
  - `offset`: Generated offset tensor, shape `(B, 18, H, W)`
  - `mu`: Predicted mean values, shape `(B, 18, H, W)`
  - `sigma`: Predicted standard deviation values, shape `(B, 18, H, W)`

### DeformASPP
- **Input:**
  - `x`: Feature map of shape `(B, C, H, W)`

- **Output:**
  - `features`: Enhanced feature map, shape `(B, out_channels, H, W)`
  - `offset_list`: List of offsets from each step
  - `mu_list`: List of mean values from each step
  - `sigma_list`: List of standard deviation values from each step

## Architecture Details

### Bayesian Offset Prediction
The module predicts offsets for 3×3 deformable convolution kernels. Each spatial location has 2 offset values (x, y) for 9 sampling points, resulting in 18 channels total.

### Progressive Accumulation
Offsets are accumulated across layers using a learned scaling factor:
```
offset = current_offset + scale_factor × previous_offset
```

### Multi-Scale Fusion
DeformASPP fuses features from multiple scales:
- Progressive deformable features (steps × in_channels)
- Global context features (in_channels)
- Total fused: (steps + 1) × in_channels → out_channels

## Citation

If you use this code in your research, please cite the RoDD paper:

```
Song J, Liu Y, Xu J, et al. TARD: An Efficient Adaptive Decoder Mechanism with Progressive Offset Accumulation and Cascaded Adaptive Receptive Field Expansion[C]//International Conference on Neural Information Processing. Singapore: Springer Nature Singapore, 2025: 464-479.
```

## License

This code is released under the MIT License.
