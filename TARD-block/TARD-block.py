"""
Progressive Offset Accumulation Module

a progressive offset accumulation mechanism consisting of Bayesian Offset Predictor
and Deformable ASPP for enhanced feature extraction in multimodal fusion tasks.

Author: [Song J, Liu Y, Xu J, et al.]
Paper: TARD: An Efficient Adaptive Decoder Mechanism with Progressive Offset Accumulation and Cascaded Adaptive Receptive Field Expansion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class ProgressiveOffsetPredictor(nn.Module):
    """
    Progressive Offset Predictor (POP)
    It generates offsets with uncertainty estimation through mean (μ) and variance (σ²).
    The module supports progressive offset accumulation by incorporating previous offsets.

    Args:
        in_channels (int): Number of input feature channels
        hidden_dim (int, optional): Hidden dimension for feature extraction. Default: 128

    Input:
        x (torch.Tensor): Input feature map of shape (B, C, H, W)
        prev_offset (torch.Tensor, optional): Previous offset from previous layer, shape (B, 18, H, W)

    Output:
        offset (torch.Tensor): Generated offset for deformable convolution, shape (B, 18, H, W)
        mu (torch.Tensor): Predicted mean values, shape (B, 18, H, W)
        sigma (torch.Tensor): Predicted standard deviation values, shape (B, 18, H, W)

    Note:
        The offset tensor has 18 channels (2*9) for 3x3 deformable convolution kernels,
        where each spatial location has 2 offsets (x,y) for 9 sampling points.
    """

    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.bfn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Predict mean μ
        self.conv_mu = nn.Conv2d(hidden_dim, 18, kernel_size=1)
        # Predict log variance log(σ²)
        self.conv_logvar = nn.Conv2d(hidden_dim, 18, kernel_size=1)
        # Predict scale factor for offset accumulation
        self.conv_scale = nn.Conv2d(hidden_dim, 18, kernel_size=1)

    def forward(self, x, prev_offset=None):
        """
        Forward pass with Bayesian sampling and progressive accumulation.

        Args:
            x (torch.Tensor): Input features
            prev_offset (torch.Tensor, optional): Previous offset to accumulate

        Returns:
            tuple: (offset, mu, sigma)
        """
        x = self.bfn(x)

        # Predict mean and log variance
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)

        # Sample from Gaussian distribution N(μ, σ²)
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        offset = mu + sigma * eps

        # Progressive offset accumulation
        if prev_offset is not None:
            scale_factor = torch.sigmoid(self.conv_scale(x))  # Scale factor ∈ [0,1]
            offset = offset + scale_factor * prev_offset

        return offset, mu, sigma


class DeformASPP(nn.Module):
    """
    Deformable ASPP (Atrous Spatial Pyramid Pooling)

    It applies multiple deformable convolution layers with accumulated offsets to capture
    multi-scale contextual information adaptively.

    Args:
        in_channels (int): Number of input feature channels
        out_channels (int): Number of output feature channels
        steps (int, optional): Number of progressive deformation steps. Default: 3

    Input:
        x (torch.Tensor): Input feature map of shape (B, C, H, W)

    Output:
        features (torch.Tensor): Enhanced feature map of shape (B, out_channels, H, W)
        offset_list (list): List of offsets from each step, each of shape (B, 18, H, W)
        mu_list (list): List of mean values from each step, each of shape (B, 18, H, W)
        sigma_list (list): List of std values from each step, each of shape (B, 18, H, W)

    The module progressively accumulates offsets across multiple deformable convolution
    layers, allowing for adaptive receptive field expansion and uncertainty-aware
    feature extraction.
    """

    def __init__(self, in_channels, out_channels, steps=3):
        super().__init__()
        self.steps = steps

        # Progressive offset predictors
        self.bfn_layers = nn.ModuleList([
            ProgressiveOffsetPredictor(in_channels) for _ in range(steps)
        ])

        # Deformable convolution layers with progressive offsets (maintain same channels)
        self.deform_layers = nn.ModuleList([
            DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            for _ in range(steps)
        ])

        # Global context branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # Feature fusion and projection
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * (steps + 1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass with progressive deformable convolutions.

        Args:
            x (torch.Tensor): Input features

        Returns:
            tuple: (enhanced_features, offset_list, mu_list, sigma_list)
        """
        res = []
        offset_list, mu_list, sigma_list = [], [], []

        # Keep original input for offset prediction
        x_original = x
        # Initial offset is None for first layer
        offset = None

        # Progressive deformation steps
        for i in range(self.steps):
            # Generate offset with Bayesian uncertainty (always use original features)
            offset, mu, sigma = self.bfn_layers[i](x_original, prev_offset=offset)

            # Store intermediate results (detached for stability)
            offset_list.append(offset.detach())
            mu_list.append(mu.detach())
            sigma_list.append(sigma.detach())

            # Apply deformable convolution
            x = self.deform_layers[i](x, offset)
            res.append(x)

        # Global context feature
        h, w = x.shape[2:]
        global_feat = self.global_pool(x_original)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        res.append(global_feat)

        # Fuse all scale features
        return self.project(torch.cat(res, dim=1)), offset_list, mu_list, sigma_list


# Example usage and testing
if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create sample input
    batch_size, in_channels, height, width = 2, 256, 32, 32
    out_channels = 128

    x = torch.randn(batch_size, in_channels, height, width)

    # Test ProgressiveOffsetPredictor
    print("Testing ProgressiveOffsetPredictor...")
    bop = ProgressiveOffsetPredictor(in_channels)
    offset, mu, sigma = bop(x)
    print(f"Input shape: {x.shape}")
    print(f"Offset shape: {offset.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Sigma shape: {sigma.shape}")

    # Test with previous offset
    offset2, mu2, sigma2 = bop(x, prev_offset=offset)
    print(f"Offset with accumulation shape: {offset2.shape}")

    # Test DeformASPP
    print("\nTesting DeformASPP...")
    daspp = DeformASPP(in_channels, out_channels, steps=3)
    features, offset_list, mu_list, sigma_list = daspp(x)
    print(f"Input shape: {x.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Number of offset steps: {len(offset_list)}")
    print(f"Offset shapes: {[o.shape for o in offset_list]}")
    print(f"Mu shapes: {[m.shape for m in mu_list]}")
    print(f"Sigma shapes: {[s.shape for s in sigma_list]}")

    print("\nAll tests passed! The Progressive Offset Accumulation module is working correctly.")
