from dataclasses import dataclass


@dataclass
class LUTConfig:
    vec_len: int = 2
    ncentroids: int = 64
    distance_p: str = "2.0"
    keep_last_n: int = 0
    checkpoint: str = None
    # decoder layer indices to convert; None = all (subject to keep_last_n). The rest stay full precision.
    lut_layers: list[int] | None = None
    quantize_lm_head: bool = True
    # none | gs (per-token gain-shape everywhere) | rot (Hadamard everywhere) | mix (Hadamard on down_proj, gain-shape elsewhere)
    transform: str = "none"
    beta: float = 1.0  # commitment weight in lut_loss
    codebook_task_grad: bool = False  # let the task loss update centroids (in addition to lut_loss)
    assign_chunk: int = 4096  # tokens per nearest-centroid chunk

    @classmethod
    def load_from_yaml(cls, yaml_path: str):
        import yaml

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def lut_layer_set(self, nlayers: int) -> set[int]:
        layers = range(nlayers - self.keep_last_n) if self.lut_layers is None else self.lut_layers
        return {i for i in layers if 0 <= i < nlayers}


DEFAULT_CONFIG = LUTConfig()
