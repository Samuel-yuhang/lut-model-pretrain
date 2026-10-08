"""Layer sensitivity scan for choosing which decoder layers stay full precision (hybrid scheme B).

Base model: all decoder layers LUT (k-means init, or a distilled checkpoint via --ckpt). A layer is made "full
precision" by swapping in the teacher's original decoder layer. Metric: KL(teacher || hybrid) on held-out tokens.

  in     : only layer i is LUT, everything else FP          -> one-shot sensitivity  dKL_i
  out    : everything LUT except layer i                    -> KL reduction from reverting layer i
  greedy : grow the FP set one layer at a time, each round picking the layer whose revert lowers KL most

usage: PYTHONPATH=. python scripts/sensitivity_scan.py --lut_config configs/distill_all_mix.yaml \
         [--ckpt checkpoints/F2_all_mix_seq/best] --modes in,out,greedy --nfp 7 --out results/sens_ptq.json
"""

import argparse
import json
import time

import torch
import yaml
from transformers import AutoModelForCausalLM

from lutneuro.config import LUTConfig
from lutneuro.distill import TokenFile, decoder, embed_and_rope
from lutneuro.models.lut_model import AutoModelForLutLM, load_from_safetensors
from lutneuro.ops.kmeans import kmeans_init


@torch.no_grad()
def make_kl(student, teacher, ids, batch):
    """Returns kl(fp_set): KL(teacher || hybrid) where layers in fp_set use the teacher's decoder layer."""
    L = len(decoder(teacher).layers)
    ref = []  # teacher final hidden states, computed once
    for b in range(0, ids.shape[0], batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, pe = embed_and_rope(teacher, ids[b : b + batch])
            for i in range(L):
                h = decoder(teacher).layers[i](h, attention_mask=None, position_embeddings=pe)
        ref.append(h)

    @torch.no_grad()
    def kl(fp_set):
        tot, n = 0.0, 0
        for bi, b in enumerate(range(0, ids.shape[0], batch)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h, pe = embed_and_rope(teacher, ids[b : b + batch])
                for i in range(L):
                    layer = decoder(teacher).layers[i] if i in fp_set else decoder(student).layers[i]
                    h = layer(h, attention_mask=None, position_embeddings=pe)
                for j in range(h.shape[0]):
                    lp = teacher.lm_head(decoder(teacher).norm(h[j])).float().log_softmax(-1)
                    lt = teacher.lm_head(decoder(teacher).norm(ref[bi][j])).float().log_softmax(-1)
                    tot += (lt.exp() * (lt - lp)).sum().item()
                    n += lp.shape[0]
        return tot / n

    return kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--lut_config", required=True)
    ap.add_argument("--ckpt", default=None, help="distilled all-LUT checkpoint; default: k-means PTQ init")
    ap.add_argument("--data_dir", default="data/fineweb10bt")
    ap.add_argument("--first_seq", type=int, default=512, help="held-out seqs [first, first+n): disjoint from final eval")
    ap.add_argument("--nseq", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--modes", default="in,out,greedy")
    ap.add_argument("--nfp", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)

    teacher = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    with open(args.lut_config) as f:
        cfg = LUTConfig(**{**yaml.safe_load(f), "lut_layers": None})
    student = AutoModelForLutLM.from_pretrained(args.model, lut_config=cfg, dtype=torch.float32).cuda().eval()
    if args.ckpt:
        load_from_safetensors(student, args.ckpt)
    else:
        kmeans_init(student, TokenFile(f"{args.data_dir}/calib.bin", 2048).get(range(64)).cuda())
    ids = TokenFile(f"{args.data_dir}/heldout.bin", 2048).get(range(args.first_seq, args.first_seq + args.nseq)).cuda()
    kl = make_kl(student, teacher, ids, args.batch)
    L = len(decoder(teacher).layers)
    allset = set(range(L))
    t0 = time.time()
    res = {"base": args.ckpt or "kmeans_ptq", "lut_config": args.lut_config, "kl_all_lut": kl(set())}
    print(json.dumps(res), flush=True)
    modes = args.modes.split(",")
    if "in" in modes:  # only layer i quantized
        res["in"] = [kl(allset - {i}) for i in range(L)]
        print("in", [round(x, 4) for x in res["in"]], f"{time.time() - t0:.0f}s", flush=True)
    if "out" in modes:  # all quantized except layer i
        res["out"] = [kl({i}) for i in range(L)]
        print("out", [round(x, 4) for x in res["out"]], f"{time.time() - t0:.0f}s", flush=True)
    if "greedy" in modes:
        fp, trace = set(), []
        for _ in range(args.nfp):
            scores = {i: kl(fp | {i}) for i in sorted(allset - fp)}
            best = min(scores, key=scores.get)
            fp.add(best)
            trace.append({"add": best, "kl": scores[best], "scores": scores})
            print("greedy add", best, "kl", round(scores[best], 4), "fp", sorted(fp), f"{time.time() - t0:.0f}s", flush=True)
        res["greedy"] = trace
        res["greedy_fp"] = sorted(fp)
    if "in" in res:
        res["topk_in_fp"] = sorted(sorted(range(L), key=lambda i: -res["in"][i])[: args.nfp])
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
