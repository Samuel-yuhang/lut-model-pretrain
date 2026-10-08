import glob
import os

import torch.nn as nn
from safetensors.torch import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from lutneuro.config import DEFAULT_CONFIG, LUTConfig
from lutneuro.ops.lut_linear import LUTLinear


def _transform_flags(config: LUTConfig, name: str) -> tuple[bool, bool]:
    """(rotate, gain_shape) for a linear called `name` under config.transform."""
    t = config.transform
    if t == "none":
        return False, False
    if t == "gs":
        return False, True
    if t == "rot":
        return True, False
    if t == "mix":
        return (True, False) if name == "down_proj" else (False, True)
    raise ValueError(f"unknown transform {t}")


def replace_linear_with_lutlinear(module: nn.Module, config: LUTConfig, name_parts: list[str] = None):
    if name_parts is None:
        name_parts = []

    names = [name for name, _ in module.named_children()]
    is_layer_list = bool(names) and names[0].isdigit()
    lut_layers = config.lut_layer_set(len(names)) if is_layer_list else None

    for name, child in module.named_children():
        # Decoder layers outside the LUT set stay full precision
        if is_layer_list and int(name) not in lut_layers:
            continue

        new_name_parts = name_parts + [name]
        layer_name = ".".join(new_name_parts)

        if isinstance(child, nn.Linear):
            if layer_name == "lm_head" and not config.quantize_lm_head:
                continue
            rotate, gain_shape = _transform_flags(config, name)
            lut_linear = LUTLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                name=layer_name,
                config=config,
                device=child.weight.device,
                dtype=child.weight.dtype,
                bias=child.bias is not None,
                rotate=rotate,
                gain_shape=gain_shape,
            )
            lut_linear.weight.data.copy_(child.weight.clone())
            lut_linear.bias.data.copy_(child.bias.clone()) if child.bias is not None else None
            setattr(module, name, lut_linear)
        else:
            replace_linear_with_lutlinear(child, config, new_name_parts)


def load_from_safetensors(model: nn.Module, checkpoint: str):
    safetensors_files = glob.glob(os.path.join(checkpoint, "*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No .safetensors files found in {checkpoint}")
    state_dict = {}
    for file_path in safetensors_files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    if getattr(model.config, "tie_word_embeddings", False) and "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    model.load_state_dict(state_dict)


class AutoModelForLutLM:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        lut_config: LUTConfig = DEFAULT_CONFIG,
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
        model.eval()

        replace_linear_with_lutlinear(model, lut_config)

        if lut_config.checkpoint is not None:
            load_from_safetensors(model, lut_config.checkpoint)
        return model
