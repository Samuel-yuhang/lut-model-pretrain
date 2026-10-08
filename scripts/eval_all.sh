#!/usr/bin/env bash
# Full evaluation (layer-wise + PPL + lm-eval) of each final checkpoint, one GPU per checkpoint, in parallel.
cd "$(dirname "$0")/.."
PY=${PY:-python}  # interpreter with torch, transformers, lm-eval, pyarrow, matplotlib
mkdir -p results logs
gpu=0
for spec in "$@"; do  # name=checkpoint_dir
  name=${spec%%=*}; ckpt=${spec#*=}
  PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" scripts/eval_layerwise.py \
    --ckpt "$ckpt" --lm_eval --out "results/$name.json" > "logs/eval_$name.log" 2>&1 &
  gpu=$((gpu + 1))
done
wait
