import torch

from lutneuro.ops.lut_linear import LUTLinear


@torch.no_grad()
def batched_kmeans(x: torch.Tensor, c: int, iters: int = 30, seed: int = 0) -> torch.Tensor:
    """Batched Lloyd k-means. x: (K, N, v) float32 -> centroids (K, c, v). Empty clusters are re-seeded."""
    K, N, v = x.shape
    kc = max(1, int(1.5e9 // (N * c)))
    if K > kc:  # bound the (K, N, c) distance tensor
        return torch.cat([batched_kmeans(x[i : i + kc], c, iters, seed + i) for i in range(0, K, kc)])
    g = torch.Generator(device=x.device).manual_seed(seed)
    C = torch.gather(x, 1, torch.randint(0, N, (K, c), device=x.device, generator=g).unsqueeze(-1).expand(-1, -1, v))
    for _ in range(iters):
        a = ((C * C).sum(-1).unsqueeze(1) - 2 * torch.einsum("knv,kcv->knc", x, C)).argmin(-1)
        S = torch.zeros_like(C).scatter_add_(1, a.unsqueeze(-1).expand(-1, -1, v), x)
        n = torch.zeros(K, c, device=x.device).scatter_add_(1, a, torch.ones_like(a, dtype=torch.float32))
        empty = (n == 0).unsqueeze(-1)
        reseed = torch.gather(x, 1, torch.randint(0, N, (K, c), device=x.device, generator=g).unsqueeze(-1).expand(-1, -1, v))
        C = torch.where(empty, reseed, S / n.clamp(min=1).unsqueeze(-1))
    return C


@torch.no_grad()
def kmeans_init(model: torch.nn.Module, calib_ids: torch.Tensor, nsamp: int = 32768, batch: int = 4, iters: int = 30):
    """Initialize every LUTLinear codebook with k-means on its (transformed) full-precision input activations.

    calib_ids: (nseq, seqlen) token ids. All LUTLinear layers are bypassed while collecting, so every layer sees
    the activations of the unquantized network."""
    luts = {n: m for n, m in model.named_modules() if isinstance(m, LUTLinear)}
    per_call = max(1, nsamp // max(1, calib_ids.shape[0] // batch))
    samples = {n: [] for n in luts}

    def hook(name):
        def f(mod, inp):
            z, _ = mod.transform(inp[0].reshape(-1, mod.in_features))
            idx = torch.randperm(z.shape[0], device=z.device)[:per_call]
            samples[name].append(z[idx].float())

        return f

    for m in luts.values():
        m.bypass = True
    hs = [m.register_forward_pre_hook(hook(n)) for n, m in luts.items()]
    dev = next(model.parameters()).device
    for i in range(0, calib_ids.shape[0], batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(calib_ids[i : i + batch].to(dev))
    for h in hs:
        h.remove()
    for i, (n, m) in enumerate(luts.items()):
        m.bypass = False
        z = torch.cat(samples.pop(n))[:nsamp]
        x = z.reshape(-1, m.ncodebooks, m.vec_len).transpose(0, 1).contiguous()
        C = batched_kmeans(x, m.ncentroids, iters, seed=i)
        m.centroids.weight.copy_(C.reshape(m.ncodebooks, -1).to(m.centroids.weight.dtype))
