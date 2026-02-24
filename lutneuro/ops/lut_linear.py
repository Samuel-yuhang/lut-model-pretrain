import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from lutneuro.config import LUTConfig


class LUTLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LUTConfig,
        bias: bool = True,
        name: str = "",
        device=None,
        dtype=None,
        **factory_kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert in_features % config.vec_len == 0
        self.ncodebooks = in_features // config.vec_len
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.ncentroids = config.ncentroids
        self.vec_len = config.vec_len
        self.distance_p = config.distance_p.lower()

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs, requires_grad=True))
        self.centroids = nn.Embedding(self.ncodebooks, self.ncentroids * self.vec_len, **factory_kwargs)
        # self.centroids = Parameter(
        #     torch.empty((self.ncodebooks, self.ncentroids, self.vec_len), **factory_kwargs),
        #     requires_grad=False,
        # )

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=True)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5e-2)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reshape x to (batch_size, seq_len, in_features)
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x_reshaped = x.reshape(-1, self.in_features)
        centroids = self.centroids.weight.reshape(self.ncodebooks, self.ncentroids, self.vec_len)

        x_vec = x_reshaped.reshape(-1, self.ncodebooks, self.vec_len).transpose(
            0, 1
        )  # (ncodebooks, batch * seq_len, vec_len)
        dists = torch.cdist(x_vec, centroids, p=2)  # (ncodebooks, batch * seq_len, ncentroids)
        indices = dists.argmin(dim=-1)  # (ncodebooks, batch * seq_len)
        quantized_x = (
            torch.gather(centroids, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, self.vec_len))
            .transpose(0, 1)
            .reshape(-1, self.in_features)
        )
        quantized_x = quantized_x.reshape(x.shape)
        self.lut_loss = F.mse_loss(quantized_x, x.detach()) + F.mse_loss(quantized_x.detach(), x)
        quantized_x = x + (quantized_x - x).detach() if self.training else quantized_x

        return F.linear(quantized_x, self.weight, self.bias)
