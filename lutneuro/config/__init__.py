from dataclasses import dataclass


@dataclass
class LUTConfig:
    vec_len: int = 2
    ncentroids: int = 64
    distance_p: str = "2.0"
    keep_last_n: int = 0
    checkpoint: str = None

    @classmethod
    def load_from_yaml(cls, yaml_path: str):
        import yaml

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


DEFAULT_CONFIG = LUTConfig()
