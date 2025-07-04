import math
import torch
import torch.nn.functional as F
import torch.nn as nn

# from .conv import ConvModule
# from ..builder import MODELS


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4, transpose=False):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)), requires_grad=True
        )
        self.drop_prob = drop_prob
        self.transpose = (
            transpose  # if False, the input is B,C,T, otherwise, the input is B,T,C
        )

    def forward(self, x):
        if self.transpose:
            x = x.transpose(1, 2)
            x = drop_path(self.scale * x, self.drop_prob, self.training)
            return x.transpose(1, 2)
        else:
            return drop_path(self.scale * x, self.drop_prob, self.training)

    def __repr__(self):
        return f"{self.__class__.__name__}(drop_prob={self.drop_prob})"
