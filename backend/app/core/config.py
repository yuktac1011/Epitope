import yaml
from pydantic import BaseModel, Field
from typing import List, Literal

class DPSettings(BaseModel):
    enabled: bool = False
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0

class AttackSettings(BaseModel):
    enabled: bool = False
    type: str = "label_flip"
    attacker_fraction: float = 0.1

class ExperimentConfig(BaseModel):
    method: Literal["local", "fedavg", "fedprox"]
    model: Literal["mlp", "hybrid"]
    bank_ids: List[int]
    rounds: int = 5
    local_epochs: int = 1
    seeds: List[int] = [42, 43, 44]
    aggregation_strategy: str = "FedAvg"
    dp: DPSettings = Field(default_factory=DPSettings)
    attack: AttackSettings = Field(default_factory=AttackSettings)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
