import math
import zlib

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from lutneuro.config import LUTConfig


def _paley_hadamard(q: int) -> torch.Tensor:
    """Paley construction I: Hadamard matrix of order q + 1 for prime q = 3 (mod 4)."""
    residues = {(i * i) % q for i in range(1, q)}
    chi = [0] + [1 if i in residues else -1 for i in range(1, q)]
    Q = torch.tensor([[chi[(j - i) % q] for j in range(q)] for i in range(q)], dtype=torch.float64)
    H = torch.ones(q + 1, q + 1, dtype=torch.float64)
    H[1:, 1:] = Q + torch.eye(q, dtype=torch.float64)
    H[1:, 0] = -1
    return H


def hadamard(n: int) -> torch.Tensor:
    """Unnormalized Hadamard matrix of order n = 2^k or 12 * 2^k (Kronecker of Paley-12 and Sylvester)."""
    base, m = torch.ones(1, 1, dtype=torch.float64), n
    if n % 12 == 0 and (n // 12) & (n // 12 - 1) == 0:
        base, m = _paley_hadamard(11), n // 12
    assert m & (m - 1) == 0, f"no Hadamard construction for n={n}"
    H2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    H = base
    while H.shape[0] < n:
        H = torch.kron(H, H2)
    return H


def random_hadamard(n: int, seed: int) -> torch.Tensor:
    """Orthogonal randomized Hadamard D H / sqrt(n) (deployable as a fast online transform, as in QuaRot/SpinQuant R4)."""
    g = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n,), generator=g, dtype=torch.float64) * 2 - 1
    return (signs[:, None] * hadamard(n) / math.sqrt(n)).float()


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
        rotate: bool = False,
        gain_shape: bool = False,
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
        self.distance_p = float(config.distance_p)
        self.beta = config.beta
        self.codebook_task_grad = config.codebook_task_grad
        self.assign_chunk = config.assign_chunk
        self.gain_shape = gain_shape
        self.bypass = False  # True: behave as plain nn.Linear (used while collecting calibration activations)
        self.track_usage = False  # count centroid assignments and keep recent inputs (dead-centroid restart)
        self.usage = None
        self.recent = None

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs, requires_grad=True))
        self.centroids = nn.Embedding(self.ncodebooks, self.ncentroids * self.vec_len, **factory_kwargs)
        if rotate:
            self.register_buffer("rotation", random_hadamard(in_features, seed=zlib.crc32(name.encode())).to(device))
        else:
            self.rotation = None

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
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
            f"rotate={self.rotation is not None}, gain_shape={self.gain_shape}, p={self.distance_p}"
        )

    def transform(self, x: torch.Tensor):
        """Pre-quantization transform: optional rotation, then optional per-token RMS (gain-shape) normalization."""
        scale = None
        if self.rotation is not None:
            x = x @ self.rotation.to(x.dtype)
        if self.gain_shape:
            scale = x.float().pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6).to(x.dtype)
            x = x / scale
        return x, scale

    def inverse_transform(self, x: torch.Tensor, scale):
        if scale is not None:
            x = x * scale
        if self.rotation is not None:
            x = x @ self.rotation.to(x.dtype).T
        return x

    @torch.no_grad()
    def assign(self, x_vec: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """Nearest centroid per sub-vector, chunked over tokens. x_vec: (ncodebooks, n, v) -> (ncodebooks, n)."""
        out = []
        c = centroids.float()
        cn = (c * c).sum(-1).unsqueeze(1)
        for i in range(0, x_vec.shape[1], self.assign_chunk):
            xi = x_vec[:, i : i + self.assign_chunk].float()
            if self.distance_p == 2.0:
                d = cn - 2 * torch.einsum("knv,kcv->knc", xi, c)
            elif math.isinf(self.distance_p):
                d = (xi.unsqueeze(2) - c.unsqueeze(1)).abs().amax(-1)
            else:
                d = torch.cdist(xi, c, p=self.distance_p)
            out.append(d.argmin(dim=-1))
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def _record(self, z_vec: torch.Tensor, indices: torch.Tensor):
        if self.usage is None:
            self.usage = torch.zeros(self.ncodebooks, self.ncentroids, device=indices.device)
        self.usage.scatter_add_(1, indices, torch.ones_like(indices, dtype=self.usage.dtype))
        pick = torch.randint(0, z_vec.shape[1], (256,), device=z_vec.device)
        self.recent = z_vec[:, pick].detach().float()  # (ncodebooks, 256, vec_len)

    @torch.no_grad()
    def restart_dead(self, min_count: float = 1.0) -> float:
        """Re-seed centroids used fewer than min_count times since the last call with recent inputs.
        Returns the dead fraction."""
        if self.usage is None or self.recent is None:
            return 0.0
        dead = self.usage < min_count
        frac = dead.float().mean().item()
        if dead.any():
            C = self.centroids.weight.view(self.ncodebooks, self.ncentroids, self.vec_len)
            pick = torch.randint(0, self.recent.shape[1], (self.ncodebooks, self.ncentroids), device=C.device)
            cand = torch.gather(self.recent, 1, pick.unsqueeze(-1).expand(-1, -1, self.vec_len)).to(C.dtype)
            C.copy_(torch.where(dead.unsqueeze(-1), cand, C))
        self.usage.zero_()
        return frac

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        x = x.unsqueeze(1) if x.dim() == 2 else x
        z, scale = self.transform(x.reshape(-1, self.in_features))
        centroids = self.centroids.weight.reshape(self.ncodebooks, self.ncentroids, self.vec_len)

        z_vec = z.reshape(-1, self.ncodebooks, self.vec_len).transpose(0, 1)  # (ncodebooks, tokens, vec_len)
        indices = self.assign(z_vec, centroids)  # (ncodebooks, tokens)
        if self.track_usage:
            self._record(z_vec, indices)
        quantized = (
            torch.gather(centroids, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, self.vec_len))
            .transpose(0, 1)
            .reshape(-1, self.in_features)
        ).to(z.dtype)
        self.lut_loss = F.mse_loss(quantized, z.detach()) + self.beta * F.mse_loss(quantized.detach(), z)
        if self.training:
            # STE. codebook_task_grad=True also routes the task-loss gradient into the centroids.
            quantized = quantized + (z - z.detach()) if self.codebook_task_grad else z + (quantized - z).detach()
        quantized = self.inverse_transform(quantized, scale).reshape(x.shape)
        return F.linear(quantized, self.weight, self.bias)
