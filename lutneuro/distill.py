"""Shared helpers for layer-/group-wise distillation of LUT models against a full-precision teacher."""

import math

import numpy as np
import torch
import torch.nn.functional as F


class TokenFile:
    """Packed uint32 token file viewed as (nseq, seqlen)."""

    def __init__(self, path: str, seqlen: int):
        data = np.memmap(path, dtype=np.uint32, mode="r")
        self.nseq = data.size // seqlen
        self.view = data[: self.nseq * seqlen].reshape(self.nseq, seqlen)

    def get(self, idx) -> torch.Tensor:
        return torch.from_numpy(self.view[np.sort(np.asarray(idx))].astype(np.int64))


def decoder(model):
    return model.model


def embed_and_rope(model, ids: torch.Tensor):
    m = decoder(model)
    h = m.embed_tokens(ids)
    pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    return h, m.rotary_emb(h, pos)


def run_layer(model, i: int, h: torch.Tensor, pe):
    # attention_mask=None -> causal SDPA inside the attention module
    return decoder(model).layers[i](h, attention_mask=None, position_embeddings=pe)


@torch.no_grad()
def teacher_states(teacher, ids: torch.Tensor):
    """Hidden states [h_0 (embeddings), h_1, ..., h_L] of the teacher, plus rotary embeddings."""
    h, pe = embed_and_rope(teacher, ids)
    hs = [h]
    for i in range(len(decoder(teacher).layers)):
        h = run_layer(teacher, i, h, pe)
        hs.append(h)
    return hs, pe


@torch.no_grad()
def channel_rms(teacher, ids: torch.Tensor, batch: int = 4, floor_frac: float = 0.1) -> list[torch.Tensor]:
    """Per-layer, per-channel RMS of teacher hidden states h_1..h_L (used to normalize the MSE: Qwen's residual
    stream has massive-activation channels that would otherwise dominate). Floored at floor_frac * median."""
    acc = None
    n = 0
    for i in range(0, ids.shape[0], batch):
        hs, _ = teacher_states(teacher, ids[i : i + batch])
        sq = [h.float().pow(2).sum((0, 1)) for h in hs[1:]]
        acc = sq if acc is None else [a + s for a, s in zip(acc, sq, strict=True)]
        n += hs[0].shape[0] * hs[0].shape[1]
    rms = [(a / n).sqrt() for a in acc]
    return [r.clamp(min=floor_frac * r.median()) for r in rms]


def nmse(y: torch.Tensor, t: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
    return ((y.float() - t.float()) / rms).pow(2).mean()


@torch.no_grad()
def layer_metrics(y: torch.Tensor, t: torch.Tensor, rms: torch.Tensor) -> dict:
    y, t = y.float(), t.float()
    return {
        "nmse": nmse(y, t, rms).item(),
        "rel_mse": ((y - t).pow(2).sum() / t.pow(2).sum()).item(),
        "cos": F.cosine_similarity(y, t, dim=-1).mean().item(),
    }


@torch.no_grad()
def evaluate(student, teacher, ids: torch.Tensor, rms: list[torch.Tensor], batch: int = 2) -> dict:
    """Per-layer teacher-forced and free-running metrics, plus PPL / KL / top-1 agreement vs the teacher."""
    L = len(decoder(student).layers)
    tf = [dict(nmse=0.0, rel_mse=0.0, cos=0.0) for _ in range(L)]
    fr = [dict(nmse=0.0, rel_mse=0.0, cos=0.0) for _ in range(L)]
    nll_s = nll_t = kl = agree = 0.0
    ntok = 0
    nb = 0
    was_training = student.training
    student.eval()
    for b in range(0, ids.shape[0], batch):
        x = ids[b : b + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs, pe = teacher_states(teacher, x)
            h = hs[0]
            for i in range(L):
                for k, v in layer_metrics(run_layer(student, i, hs[i], pe), hs[i + 1], rms[i]).items():
                    tf[i][k] += v
                h = run_layer(student, i, h, pe)
                for k, v in layer_metrics(h, hs[i + 1], rms[i]).items():
                    fr[i][k] += v
            for j in range(x.shape[0]):  # logits one sequence at a time (vocab is 152k)
                ls = student.lm_head(decoder(student).norm(h[j : j + 1])).float()[0, :-1]
                lt = teacher.lm_head(decoder(teacher).norm(hs[-1][j : j + 1])).float()[0, :-1]
                tgt = x[j, 1:]
                nll_s += F.cross_entropy(ls, tgt, reduction="sum").item()
                nll_t += F.cross_entropy(lt, tgt, reduction="sum").item()
                lps, lpt = ls.log_softmax(-1), lt.log_softmax(-1)
                kl += (lpt.exp() * (lpt - lps)).sum().item()
                agree += (ls.argmax(-1) == lt.argmax(-1)).sum().item()
                ntok += tgt.numel()
        nb += 1
    student.train(was_training)
    avg = lambda rows: [{k: v / nb for k, v in r.items()} for r in rows]  # noqa: E731
    return {
        "ppl_student": math.exp(nll_s / ntok),
        "ppl_teacher": math.exp(nll_t / ntok),
        "kl": kl / ntok,
        "top1_agree": agree / ntok,
        "teacher_forced": avg(tf),
        "free_running": avg(fr),
    }
