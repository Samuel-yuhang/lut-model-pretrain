"""Final evaluation of a distilled (hybrid) LUT checkpoint against the full-precision teacher.

  - per-layer teacher-forced / free-running nmse, rel-MSE, cosine on FineWeb held-out
  - FineWeb held-out PPL, KL(teacher || student), top-1 agreement
  - wikitext2 test PPL (seqlen 2048, same protocol as the earlier PTQ study)
  - optional lm-eval zero-shot tasks

usage: PYTHONPATH=. python scripts/eval_layerwise.py --ckpt checkpoints/<exp>/final [--lm_eval] --out results/<exp>.json
       (--ckpt fp evaluates the teacher itself)
"""

import argparse
import glob
import json
import math
import os

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lutneuro.config import LUTConfig
from lutneuro.distill import TokenFile, channel_rms, evaluate
from lutneuro.models.lut_model import AutoModelForLutLM

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "lambada_openai"]


@torch.no_grad()
def wikitext2_ppl(model, tok, seqlen=2048):
    root = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1"))[0]
    text = pq.read_table(glob.glob(f"{root}/test-*.parquet")[0]).column("text").to_pylist()
    ids = tok("\n\n".join(text), return_tensors="pt").input_ids[0]
    nll, cnt = 0.0, 0
    for i in range(ids.numel() // seqlen):
        x = ids[i * seqlen : (i + 1) * seqlen].unsqueeze(0).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nll += model(x, labels=x).loss.item() * (seqlen - 1)
        cnt += seqlen - 1
    return math.exp(nll / cnt)


def lm_eval_scores(model, tok, tasks, batch_size):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, dtype="float32")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        res = simple_evaluate(model=lm, tasks=tasks, log_samples=False)
    out = {}
    for t, r in res["results"].items():
        key = "acc_norm,none" if "acc_norm,none" in r else "acc,none"
        out[t] = r.get(key)
        if t == "lambada_openai":
            out["lambada_ppl"] = r.get("perplexity,none")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data/fineweb10bt")
    ap.add_argument("--heldout_seqs", type=int, default=256)
    ap.add_argument("--lm_eval", action="store_true")
    ap.add_argument("--lm_eval_batch", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    if args.ckpt == "fp":
        student, cfg = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).cuda().eval(), None
    else:
        cfg = LUTConfig.load_from_yaml(f"{args.ckpt}/lut_config.yaml")
        cfg.checkpoint = args.ckpt
        student = AutoModelForLutLM.from_pretrained(args.model, lut_config=cfg, dtype=torch.float32).cuda().eval()

    calib = TokenFile(f"{args.data_dir}/calib.bin", 2048)
    rms = channel_rms(teacher, calib.get(range(16)).cuda())
    heldout = TokenFile(f"{args.data_dir}/heldout.bin", 2048)
    res = {"ckpt": args.ckpt, "lut_layers": None if cfg is None else sorted(cfg.lut_layer_set(len(student.model.layers))),
           "transform": None if cfg is None else cfg.transform}
    res.update(evaluate(student, teacher, heldout.get(range(min(args.heldout_seqs, heldout.nseq))).cuda(), rms))
    res["wikitext2_ppl"] = wikitext2_ppl(student, tok)
    print(json.dumps({k: v for k, v in res.items() if k not in ("teacher_forced", "free_running")}), flush=True)
    if args.lm_eval:
        del teacher
        torch.cuda.empty_cache()
        res["lm_eval"] = lm_eval_scores(student, tok, TASKS, args.lm_eval_batch)
        print(json.dumps(res["lm_eval"]), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
