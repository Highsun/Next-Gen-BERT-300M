"""Strict, serializable experiment configuration."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import yaml


@dataclass
class Config:
    model_name: str = "jhu-clsp/mmBERT-base"
    revision: str = "main"
    variant: str = "B0"
    task: str = "mlm"
    backend: str = "tiled"
    kept_global: list[int] = field(default_factory=lambda: [0, 9, 21])
    radius: int = 64
    block_size: int = 256
    tile_size: int = 128
    projection_dim: int = 128
    query_global: bool = False
    query_length: int = 64
    max_length: int = 8192
    alpha: float = 0.5
    representation: str = "late"
    num_labels: int = 2
    label_names: list[str] = field(default_factory=list)
    seed: int = 17
    precision: str = "bf16"
    lora_rank: int = 16
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    accumulation: int = 8
    max_steps: int = 1000
    warmup_fraction: float = 0.05
    transition_fraction: float = 0.1
    mask_probability: float = 0.15
    span_batch_probability: float = 0.0
    temperature: float = 0.05
    checkpointing: bool = True
    profile_pipeline: bool = False
    length_curriculum: bool = True
    max_estimated_flops: float = 0.0
    max_seconds: float = 0.0
    save_every: int = 100
    keep_checkpoints: int = 2
    log_every: int = 10
    train_file: str = "data/train.jsonl"
    output_dir: str = "runs/B0"

    def validate(self):
        if self.variant not in {"B0", "L3", "M1", "M2", "G2", "C2"}:
            raise ValueError("Unknown architecture variant")
        if self.task not in {"mlm", "classification", "ner", "qa", "retrieval", "rerank"}:
            raise ValueError("Unknown task")
        if self.backend not in {"tiled", "reference"} or self.representation not in {"single", "late", "independent"}:
            raise ValueError("Unknown attention backend or representation")
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError("Supported precision: bf16, fp32")
        for key in ("block_size", "tile_size", "projection_dim", "query_length", "max_length", "num_labels",
                    "accumulation", "max_steps", "save_every", "keep_checkpoints", "log_every"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.max_length > 8192 or self.query_length + 4 >= self.max_length:
            raise ValueError("Context/query budget invalid")
        if self.radius < 0 or self.lora_rank < 0:
            raise ValueError("Negative radius/rank")
        for key in ("alpha", "warmup_fraction", "transition_fraction", "span_batch_probability"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"Invalid fraction {key}")
        if not 0 < self.mask_probability < 1 or self.temperature <= 0:
            raise ValueError("Invalid corruption/temperature")
        if self.query_global and self.task not in {"qa", "rerank"}:
            raise ValueError("Query-global is only for paired online tasks")
        if self.label_names and len(self.label_names) != self.num_labels:
            raise ValueError("label_names must match num_labels")
        if self.learning_rate <= 0 or min(self.max_seconds, self.max_estimated_flops) < 0:
            raise ValueError("Invalid optimizer or budget")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        data = yaml.safe_load(Path(path).read_text()) if str(path).endswith(('.yaml', '.yml')) else json.loads(Path(path).read_text())
        return cls(**data).validate()
