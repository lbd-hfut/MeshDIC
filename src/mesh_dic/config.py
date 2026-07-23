from dataclasses import dataclass


@dataclass
class MeshdicConfig:
    subset_radius: int = 31
    search_radius: int = 20
    max_iterations: int = 30
    cutoff_diffnorm: float = 1e-4
    lambda_reg: float = 1e-6
    bcoef_border: int = 3


DEFAULT_CONFIG = MeshdicConfig()


def config_from_dict(d: dict) -> MeshdicConfig:
    cfg = MeshdicConfig()
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
