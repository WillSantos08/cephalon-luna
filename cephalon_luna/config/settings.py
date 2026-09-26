import os
import yaml
import torch
from dataclasses import dataclass

from config.model_config import ModelConfig
from config.train_config import (
    TrainingConfig,
    SchedulerConfig,
    GenerationConfig,
    CheckpointConfig,
    LoggingConfig,
    PathsConfig,
    HardwareConfig,
)


@dataclass
class LunaConfig:
    paths:      PathsConfig
    data_aug:   bool
    val_ratio:  float
    model:      ModelConfig
    training:   TrainingConfig
    generation: GenerationConfig
    checkpoint: CheckpointConfig
    logging:    LoggingConfig
    hardware:   HardwareConfig

    def resolve_device(self) -> torch.device:
        d = self.hardware.device
        if d == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(d)

    def ensure_dirs(self):
        dirs = [
            self.paths.raw_data,
            self.paths.processed_data,
            self.paths.tokenizer.rsplit("/", 1)[0],
            self.paths.latest_model.rsplit("/", 1)[0],
            self.paths.checkpoints,
            self.paths.logs,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


def _find_config(filename: str = "config.yaml") -> str:
    """
    Procura o config.yaml subindo os diretórios
    a partir do arquivo atual até encontrar.
    """
    # 1. Caminho passado diretamente
    if os.path.exists(filename):
        return filename

    # 2. Subir diretórios até encontrar
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        candidate = os.path.join(current, filename)
        if os.path.exists(candidate):
            return candidate
        current = os.path.dirname(current)

    raise FileNotFoundError(
        f"config.yaml não encontrado.\n"
        f"Certifique-se de que o arquivo existe na raiz do projeto."
    )


def load_config(path: str = "config.yaml") -> LunaConfig:
    """Carrega e valida o config.yaml."""
    path = _find_config(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Paths
    p = raw["paths"]
    paths = PathsConfig(
        raw_data       = p["raw_data"],
        processed_data = p["processed_data"],
        tokenizer      = p["tokenizer"],
        latest_model   = p["latest_model"],
        checkpoints    = p["checkpoints"],
        logs           = p["logs"],
    )

    # Model
    m = raw["model"]
    model = ModelConfig(
        vocab_size  = m["vocab_size"],
        context_len = m["context_len"],
        d_model     = m["d_model"],
        num_heads   = m["num_heads"],
        num_layers  = m["num_layers"],
        d_ff        = m["d_ff"],
        dropout     = m["dropout"],
    )

    # Scheduler
    sc = raw["training"]["scheduler"]
    scheduler = SchedulerConfig(
        enabled       = sc["enabled"],
        type          = sc["type"],
        warmup_epochs = sc["warmup_epochs"],
    )

    # Training
    t = raw["training"]
    training = TrainingConfig(
        epochs           = t["epochs"],
        batch_size       = t["batch_size"],
        lr               = float(t["lr"]),
        grad_clip        = t["grad_clip"],
        resume           = t["resume"],
        grad_accum_steps = t.get("grad_accum_steps", 1),
        scheduler        = scheduler,
    )

    # Generation
    g = raw["generation"]
    generation = GenerationConfig(
        temperature    = g["temperature"],
        top_k          = g["top_k"],
        max_new_tokens = g["max_new_tokens"],
    )

    # Checkpoint
    ck = raw["checkpoint"]
    checkpoint = CheckpointConfig(
        save_every_n = ck["save_every_n"],
        keep_last_n  = ck["keep_last_n"],
    )

    # Logging
    lg = raw["logging"]
    logging = LoggingConfig(
        log_every_pct  = lg["log_every_pct"],
        show_samples   = lg["show_samples"],
        num_samples    = lg["num_samples"],
        sample_prompts = lg["sample_prompts"],
    )

    # Hardware
    hw = raw["hardware"]
    hardware = HardwareConfig(
        device      = hw["device"],
        seed        = hw["seed"],
        amp         = hw.get("amp", True),
        compile     = hw.get("compile", False),
        num_workers = hw.get("num_workers", 0),
        pin_memory  = hw.get("pin_memory", True),
    )

    return LunaConfig(
        paths      = paths,
        data_aug   = raw["data"]["augment"],
        val_ratio  = raw["data"]["val_ratio"],
        model      = model,
        training   = training,
        generation = generation,
        checkpoint = checkpoint,
        logging    = logging,
        hardware   = hardware,
    )