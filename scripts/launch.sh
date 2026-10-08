#!/usr/bin/env bash
# usage: scripts/launch.sh <gpus e.g. 0,1> <port> <out_name> [distill_blockwise.py args...]
set -e
cd "$(dirname "$0")/.."
gpus=$1; port=$2; name=$3; shift 3
n=$(echo "$gpus" | tr ',' '\n' | wc -l)
mkdir -p logs checkpoints
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$gpus nohup ~/mlsys/lut-llm/.venv/bin/python -m torch.distributed.run \
  --nproc_per_node "$n" --master_port "$port" scripts/distill_blockwise.py --out_dir "checkpoints/$name" "$@" \
  > "logs/$name.log" 2>&1 &
echo "$!" > "logs/$name.pid"
