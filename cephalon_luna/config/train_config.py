from dataclasses import dataclass, field
from typing import List


@dataclass
class SchedulerConfig:
    enabled:       bool = True
    type:          str  = "cosine"
    warmup_epochs: int  = 2


@dataclass
class TrainingConfig:
    epochs:            int             = 20
    batch_size:        int             = 16
    lr:                float           = 3e-4
    grad_clip:         float           = 1.0
    resume:            bool            = False
    grad_accum_steps:  int             = 1
    scheduler:         SchedulerConfig = field(
        default_factory=SchedulerConfig
    )


@dataclass
class GenerationConfig:
    temperature:    float = 0.7
    top_k:          int   = 40
    max_new_tokens: int   = 120


@dataclass
class CheckpointConfig:
    save_every_n: int = 5
    keep_last_n:  int = 3


@dataclass
class LoggingConfig:
    log_every_pct:  int       = 20
    show_samples:   bool      = True
    num_samples:    int       = 2
    sample_prompts: List[str] = field(default_factory=list)


@dataclass
class PathsConfig:
    raw_data:       str = "cephalon_luna/data/raw"
    processed_data: str = "cephalon_luna/data/processed"
    tokenizer:      str = "cephalon_luna/data/tokenizer/tokenizer.json"
    latest_model:   str = "cephalon_luna/models/latest/model.pt"
    checkpoints:    str = "cephalon_luna/models/checkpoints"
    logs:           str = "cephalon_luna/models/logs"


@dataclass
class HardwareConfig:
    device:      str  = "auto"
    seed:        int  = 42
    amp:         bool = True   # mixed precision (fp16/bf16) quando suportado
    compile:     bool = False  # torch.compile() — só vale a pena em CUDA
    num_workers: int  = 0      # workers do DataLoader (0 = processo principal)
    pin_memory:  bool = True   # acelera transferência CPU → GPU