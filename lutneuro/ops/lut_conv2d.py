import torch
import torch.nn as nn
import torch.nn.functional as F

from lutneuro.config import DEFAULT_CONFIG, LUTConfig

from .lut_linear import LUTLinear
from .utils import calc_output_shape


class LUTConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        bias: bool = True,
        config: LUTConfig = DEFAULT_CONFIG,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.linear = LUTLinear(in_channels * kernel_size[0] * kernel_size[1], out_channels, config, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        cols = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        cols = cols.permute(0, 2, 1).flatten(0, 1)
        out = self.linear(cols)
        out_h, out_w = calc_output_shape(x.shape[2:], self.kernel_size, self.stride, self.padding)
        out = out.reshape(b, out_h, out_w, self.out_channels)
        out = out.permute(0, 3, 1, 2)
        return out
