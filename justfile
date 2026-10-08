fmt:
	uvx ruff check . --fix
	uvx ruff format .

prek:
	prek run --all-files

resnet:
	uv run accelerate launch scripts/train_cnn.py

lut_resnet:
	uv run accelerate launch scripts/train_cnn.py --lut

clm:
	uv run accelerate launch scripts/run_clm_no_trainer.py \
	--model_name_or_path Qwen/Qwen3-0.6B-Base \
	--dataset_name wikitext \
	--dataset_config_name wikitext-2-raw-v1 \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--output_dir checkpoints/Qwen3-0.6B-Base/baseline/ \
	--lr_scheduler_type cosine \
	--learning_rate 5e-5 \
	--block_size 2048 \
	--num_train_epochs 20 \
	--checkpointing_steps epoch

lut_clm:
	uv run accelerate launch scripts/run_clm_no_trainer.py \
	--model_name_or_path Qwen/Qwen3-0.6B-Base \
	--dataset_name wikitext \
	--dataset_config_name wikitext-2-raw-v1 \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--output_dir checkpoints/Qwen3-0.6B-Base/lut/ \
	--lr_scheduler_type cosine \
	--learning_rate 1e-3 \
	--weight_decay 0.0 \
	--block_size 1024 \
	--num_train_epochs 20 \
	--checkpointing_steps epoch \
	--lut_config configs/clm.yaml

# set -x LOG_LEVEL WARNING
# set -x PYTHONWARNINGS ignore
# set -x TORCH_DISTRIBUTED_DEBUG OFF
# set -x NCCL_DEBUG ERROR
# set -e NCCL_ASYNC_ERROR_HANDLING
# set -x TORCH_NCCL_ASYNC_ERROR_HANDLING 1
# set -x NCCL_DEBUG_SUBSYS 0

# layer-wise distillation of the 3:1 hybrid LUT model (scheme A, see REPORT_distill.md)
fineweb:
	~/mlsys/lut-llm/.venv/bin/python scripts/prepare_fineweb.py

distill_A:
	scripts/launch.sh 0,1 29641 F1_A_mix_seq --lut_config configs/distill_A_mix.yaml --mode seq --train_tokens 10e6 --eval_every 25

eval_A:
	PYTHONPATH=. ~/mlsys/lut-llm/.venv/bin/python scripts/eval_layerwise.py --ckpt checkpoints/F1_A_mix_seq/best --lm_eval --out results/F1_A_mix_seq.json
