"""Layer-/group-wise distillation of a (hybrid) LUT model against its full-precision teacher.

Launch: torchrun --nproc_per_node 8 scripts/distill_blockwise.py --lut_config configs/distill_A_mix.yaml ...

Modes (each group = `--group_size` consecutive decoder layers, all layer outputs supervised by the teacher):
  group : input of every group is the teacher hidden state at the group start (teacher-forced, groups independent)
  layer : same with group_size 1, only LUT layers trained
  seq   : input of each group is the (detached) output of the student's previous groups -> fixes exposure bias
  e2e   : Stage 2, end-to-end: whole student chained with gradients, loss = KL(teacher || student) on the logits
          + hidden_weight * mean per-layer nmse (start from a Stage-1 checkpoint with --init_from)
"""

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.checkpoint import checkpoint
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from lutneuro.config import LUTConfig
from lutneuro.distill import TokenFile, channel_rms, evaluate, nmse, run_layer, teacher_states
from lutneuro.models.lut_model import AutoModelForLutLM, load_from_safetensors
from lutneuro.ops.kmeans import kmeans_init
from lutneuro.ops.lut_linear import LUTLinear


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--lut_config", required=True)
    ap.add_argument("--data_dir", default="data/fineweb10bt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--init_from", default=None, help="checkpoint dir to start from (skips k-means init)")
    ap.add_argument("--mode", choices=["group", "layer", "seq", "e2e"], default="group")
    ap.add_argument("--hidden_weight", type=float, default=0.0, help="e2e: weight of the per-layer hidden-state nmse")
    ap.add_argument("--kl_chunk", type=int, default=2048, help="e2e: tokens per logits/KL chunk")
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--train_fp", action="store_true", help="also train the full-precision layers (compensators)")
    ap.add_argument("--train_tokens", type=float, default=20e6)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr_codebook", type=float, default=1e-3)
    ap.add_argument("--lr_weight", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lut_loss_weight", type=float, default=0.1)
    ap.add_argument("--loss", choices=["nmse", "relmse"], default="nmse",
                    help="nmse: MSE normalized by teacher per-channel RMS; relmse: ||y - t||^2 / ||t||^2")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--calib_seqs", type=int, default=64)
    ap.add_argument("--eval_seqs", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restart_every", type=int, default=0, help="dead-centroid restart period in steps (0 = off)")
    ap.add_argument("--set", nargs="*", default=[], help="LUT config overrides key=value (yaml-parsed)")
    return ap.parse_args()


def log(path, rec):
    print(json.dumps(rec), flush=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def chunk_kl(h_s, logp_t, norm, lm_head):
    """Summed KL(teacher || student) over a chunk of tokens; student logits are recomputed in backward."""
    logp_s = lm_head(norm(h_s)).float().log_softmax(-1)
    return F.kl_div(logp_s, logp_t, log_target=True, reduction="sum")


def save(student, cfg_dict, out_dir, extra):
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.detach().contiguous().cpu() for k, v in student.state_dict().items() if k != "lm_head.weight"}
    save_file(sd, f"{out_dir}/model.safetensors")
    with open(f"{out_dir}/lut_config.yaml", "w") as f:
        yaml.safe_dump({**cfg_dict, "checkpoint": None}, f)
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(extra, f, indent=1)


def main():
    args = parse()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(args.seed)
    is_main = rank == 0
    os.makedirs(args.out_dir, exist_ok=True)
    train_log, eval_log = f"{args.out_dir}/train_log.jsonl", f"{args.out_dir}/eval_log.jsonl"

    with open(args.lut_config) as f:
        cfg_dict = yaml.safe_load(f)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg_dict[k] = yaml.safe_load(v)
    cfg = LUTConfig(**cfg_dict)

    teacher = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    teacher.requires_grad_(False)
    student = AutoModelForLutLM.from_pretrained(args.model, lut_config=cfg, dtype=torch.float32).cuda()
    nlayers = len(student.model.layers)
    lut_layers = sorted(cfg.lut_layer_set(nlayers))

    calib = TokenFile(f"{args.data_dir}/calib.bin", args.seqlen)
    calib_ids = calib.get(range(min(args.calib_seqs, calib.nseq))).cuda()
    if args.init_from:
        load_from_safetensors(student, args.init_from)
    else:
        kmeans_init(student, calib_ids)
    for t in list(student.parameters()) + list(student.buffers()):  # identical start on every rank
        dist.broadcast(t.data, 0)
    rms = channel_rms(teacher, calib_ids[:16])

    # trainable parameters
    student.requires_grad_(False)
    groups = [list(range(s, min(s + args.group_size, nlayers))) for s in range(0, nlayers, args.group_size)]
    if args.mode == "layer":
        groups = [[i] for i in lut_layers]
    train_layers = set(lut_layers) | (set(range(nlayers)) if args.train_fp else set())
    cb_params, w_params = [], []
    for i in train_layers:
        for n, p in student.model.layers[i].named_parameters():
            p.requires_grad_(True)
            (cb_params if n.endswith("centroids.weight") else w_params).append(p)
    params = cb_params + w_params
    opt = torch.optim.AdamW(
        [{"params": cb_params, "lr": args.lr_codebook}, {"params": w_params, "lr": args.lr_weight}], weight_decay=0.0
    )
    tokens_per_step = world * args.micro_batch * args.grad_accum * args.seqlen
    steps = math.ceil(args.train_tokens / tokens_per_step)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / max(steps, 1)))
    )

    train = TokenFile(f"{args.data_dir}/train.bin", args.seqlen)
    heldout = TokenFile(f"{args.data_dir}/heldout.bin", args.seqlen)
    eval_ids = heldout.get(range(min(args.eval_seqs, heldout.nseq))).cuda()
    order = torch.randperm(train.nseq, generator=torch.Generator().manual_seed(args.seed))
    meta = dict(vars(args), lut_layers=lut_layers, groups=groups, steps=steps, tokens_per_step=tokens_per_step, world=world)
    luts = [m for i in lut_layers for m in student.model.layers[i].modules() if isinstance(m, LUTLinear)]
    for m in luts:
        m.track_usage = True
    if is_main:
        log(train_log, {"event": "start", **meta, "lut_cfg": cfg_dict, "n_trainable": sum(p.numel() for p in params)})
    best = {"kl": float("inf")}

    def do_eval(step):
        if is_main:
            r = evaluate(student, teacher, eval_ids, rms)
            log(eval_log, {"step": step, "tokens": step * tokens_per_step, **r})
            log(train_log, {"step": step, "eval_ppl": r["ppl_student"], "teacher_ppl": r["ppl_teacher"], "kl": r["kl"],
                            "top1": r["top1_agree"], "fr_nmse_last": r["free_running"][-1]["nmse"]})
            if r["kl"] < best["kl"]:
                best.update(kl=r["kl"], step=step)
                save(student, cfg_dict, f"{args.out_dir}/best", {**meta, "best_step": step, "best_kl": r["kl"]})
        dist.barrier()

    do_eval(0)
    student.train()
    t0 = time.time()
    cursor = 0
    for step in range(steps):
        stats = {"kl": 0.0, "hid": 0.0} if args.mode == "e2e" else {f"g{gi}": 0.0 for gi in range(len(groups))}
        lut_l = 0.0
        for _ in range(args.grad_accum):
            b = order[(cursor + rank * args.micro_batch) % train.nseq :][: args.micro_batch]
            cursor += world * args.micro_batch
            ids = train.get(b.tolist()).cuda()
            if args.mode == "e2e":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hs, pe = teacher_states(teacher, ids)
                    x, hid = hs[0], 0.0
                    for i in range(nlayers):  # FP layers are frozen but still pass gradients to earlier LUT layers
                        x = checkpoint(run_layer, student, i, x, pe, use_reentrant=False)
                        if args.hidden_weight > 0:
                            hid = hid + nmse(x, hs[i + 1], rms[i]) / nlayers
                    xs, ht = x.flatten(0, 1), hs[-1].flatten(0, 1)
                    kl = 0.0
                    for c in range(0, xs.shape[0], args.kl_chunk):
                        with torch.no_grad():
                            logp_t = teacher.lm_head(teacher.model.norm(ht[c : c + args.kl_chunk])).float().log_softmax(-1)
                        kl = kl + checkpoint(chunk_kl, xs[c : c + args.kl_chunk], logp_t, student.model.norm,
                                             student.lm_head, use_reentrant=False)
                    kl = kl / xs.shape[0]
                    ((kl + args.hidden_weight * hid) / args.grad_accum).backward()
                stats["kl"] += kl.item() / args.grad_accum
                stats["hid"] += float(hid) / args.grad_accum
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs, pe = teacher_states(teacher, ids)
                h_prev = hs[0]
                for gi, g in enumerate(groups):
                    x = (h_prev if args.mode == "seq" else hs[g[0]]).detach()
                    loss = 0.0
                    for i in g:
                        x = run_layer(student, i, x, pe)
                        t = hs[i + 1]
                        loss = loss + (nmse(x, t, rms[i]) if args.loss == "nmse"
                                       else (x.float() - t.float()).pow(2).sum() / t.float().pow(2).sum())
                    loss = loss / len(g)
                    ll = torch.stack([m.lut_loss for i in g if i in lut_layers for m in student.model.layers[i].modules()
                                      if isinstance(m, LUTLinear)]).mean() if any(i in lut_layers for i in g) else 0.0
                    total = (loss + args.lut_loss_weight * ll) / args.grad_accum
                    if total.requires_grad:
                        total.backward()
                    stats[f"g{gi}"] += loss.item() / args.grad_accum
                    lut_l += float(ll) / args.grad_accum / len(groups)
                    h_prev = x.detach()
        for p in params:  # manual data-parallel gradient averaging
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.clip).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if args.restart_every and (step + 1) % args.restart_every == 0:
            # usage counts are per-rank; reseed on rank 0 and broadcast so codebooks stay identical
            fracs = [m.restart_dead() for m in luts]
            for m in luts:
                dist.broadcast(m.centroids.weight.data, 0)
            if is_main:
                log(train_log, {"step": step + 1, "dead_frac_mean": sum(fracs) / len(fracs), "dead_frac_max": max(fracs)})
        elif (step + 1) % 25 == 0 and is_main:  # usage monitor without restart
            fr = [(m.usage < 1).float().mean().item() for m in luts if m.usage is not None]
            log(train_log, {"step": step + 1, "dead_frac_mean": sum(fr) / max(len(fr), 1), "dead_frac_max": max(fr, default=0)})
            for m in luts:
                m.usage.zero_() if m.usage is not None else None
        if is_main and (step % args.log_every == 0 or step == steps - 1):
            loss_val = stats["kl"] + args.hidden_weight * stats["hid"] if args.mode == "e2e" else sum(stats.values()) / len(groups)
            log(train_log, {"step": step + 1, "tokens": (step + 1) * tokens_per_step, "loss": loss_val,
                            **{k: round(v, 5) for k, v in stats.items()}, "lut_loss": lut_l, "gnorm": gnorm,
                            "lr_cb": sched.get_last_lr()[0], "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                            "sec": round(time.time() - t0, 1)})
        if (step + 1) % args.eval_every == 0 or step == steps - 1:
            do_eval(step + 1)
    if is_main:
        save(student, cfg_dict, f"{args.out_dir}/final", meta)
        log(train_log, {"event": "done", "best_step": best.get("step"), "best_kl": best["kl"]})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
