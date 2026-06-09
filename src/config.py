"""Run configuration for behavioral paired-token experiments.

Every run must persist its full config (reproducibility rule in CLAUDE.md).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Semantic names on purpose: explicit names reduce the risk of purely
# lexical pattern matching (see CLAUDE.md, "Nota sul naming").
SPECIAL_TOKENS = ["[COMPRESS]", "[RECALL]", "[REASON]"]

COMPRESS_TOKEN = "[COMPRESS]"
RECALL_TOKEN = "[RECALL]"
REASON_TOKEN = "[REASON]"

LAMBDA_C_MAX = 0.1  # hard cap — higher values risk hidden-state collapse


@dataclass
class TrainConfig:
    # --- model ---
    model_name: str = "Qwen/Qwen2.5-0.5B"

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # --- data ---
    train_file: str = "data/processed/train.jsonl"
    eval_file: str = "data/processed/eval.jsonl"
    max_length: int = 1024
    # If True, cross-entropy is computed only on tokens after [RECALL]
    # (the recall target); context and filler are input-only.
    loss_on_target_only: bool = True

    # --- optimization ---
    seed: int = 42
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # --- consistency loss (OFF by default, see CLAUDE.md collapse warning) ---
    lambda_c: float = 0.0

    # --- logging / checkpoints ---
    run_name: str = "run"
    output_dir: str = "results/checkpoints"
    log_dir: str = "results/runs"
    save_steps: int = 500
    eval_steps: int = 250
    log_steps: int = 10
    use_wandb: bool = False

    # --- runtime ---
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    # Debug mode: 100 examples, 2 epochs, CPU — mandatory before any real run.
    debug: bool = False

    def __post_init__(self) -> None:
        if self.lambda_c > LAMBDA_C_MAX:
            warnings.warn(
                f"lambda_c={self.lambda_c} exceeds the collapse-safety cap "
                f"{LAMBDA_C_MAX}; clamping. See CLAUDE.md."
            )
            self.lambda_c = LAMBDA_C_MAX
        if self.debug:
            self.apply_debug()

    def apply_debug(self) -> None:
        self.epochs = 2
        self.device = "cpu"
        self.batch_size = 2
        self.grad_accum = 1
        self.save_steps = 50
        self.eval_steps = 25
        self.run_name = f"{self.run_name}-debug"

    # debug mode caps the dataset at 100 examples (used by dataset loader)
    @property
    def max_examples(self) -> int | None:
        return 100 if self.debug else None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        data = json.loads(Path(path).read_text())
        return cls(**data)
