"""Training loop: cross-entropy + optional consistency loss.

Logs (per CLAUDE.md conventions): loss_ce, loss_consistency, loss_total,
perplexity, and the variance of [COMPRESS] hidden states across the batch
(collapse detection for the consistency loss).

Usage:
    python -m src.train --debug                      # 100 examples, 2 epochs, CPU
    python -m src.train --run-name typeA-v1
    python -m src.train --config results/checkpoints/typeA-v1/config.json
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import TrainConfig
from src.dataset import make_dataloader
from src.model import setup_model_and_tokenizer
from src.utils import resolve_device, set_seed

COLLAPSE_VARIANCE_THRESHOLD = 1e-4


def gather_token_states(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """hidden: (B, T, H), positions: (B,) with -1 for missing -> (B', H)."""
    mask = positions >= 0
    if not mask.any():
        return hidden.new_zeros((0, hidden.shape[-1]))
    idx = positions[mask]
    return hidden[mask, idx]


def compute_losses(out, batch, lambda_c: float):
    """Returns (loss_total, metrics dict). Requires output_hidden_states=True."""
    loss_ce = out.loss
    h_last = out.hidden_states[-1]

    h_compress = gather_token_states(h_last, batch["compress_pos"])
    h_recall = gather_token_states(h_last, batch["recall_pos"])

    # collapse detection: variance of [COMPRESS] states across the batch
    if h_compress.shape[0] > 1:
        compress_var = h_compress.float().var(dim=0).mean().item()
    else:
        compress_var = float("nan")

    if lambda_c > 0 and h_compress.shape[0] == h_recall.shape[0] and h_compress.shape[0] > 0:
        loss_consistency = 1 - F.cosine_similarity(h_compress, h_recall, dim=-1).mean()
    else:
        loss_consistency = torch.zeros((), device=loss_ce.device)

    loss_total = loss_ce + lambda_c * loss_consistency
    metrics = {
        "loss_ce": loss_ce.item(),
        "loss_consistency": float(loss_consistency),
        "loss_total": loss_total.item(),
        "perplexity": math.exp(min(loss_ce.item(), 20)),
        "compress_hidden_variance": compress_var,
    }
    return loss_total, metrics


@torch.no_grad()
def evaluate(model, loader, device, lambda_c: float) -> dict:
    model.eval()
    totals, n = {}, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"], output_hidden_states=True)
        _, metrics = compute_losses(out, batch, lambda_c)
        for k, v in metrics.items():
            if not math.isnan(v):
                totals[k] = totals.get(k, 0.0) + v
        n += 1
    model.train()
    return {f"eval/{k}": v / max(n, 1) for k, v in totals.items()}


def save_checkpoint(model, tokenizer, cfg: TrainConfig, name: str) -> Path:
    out = Path(cfg.output_dir) / cfg.run_name / name
    if out.exists():
        shutil.rmtree(out)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    cfg.save(out / "config.json")
    return out


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    run_dir = Path(cfg.output_dir) / cfg.run_name
    cfg.save(run_dir / "config.json")  # reproducibility: config saved up front

    model, tokenizer = setup_model_and_tokenizer(cfg)
    model.to(device)
    model.print_trainable_parameters()

    train_loader = make_dataloader(cfg.train_file, tokenizer, cfg, shuffle=True)
    eval_loader = make_dataloader(cfg.eval_file, tokenizer, cfg, shuffle=False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.epochs // cfg.grad_accum)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps)

    writer = SummaryWriter(log_dir=str(Path(cfg.log_dir) / cfg.run_name))
    wandb = None
    if cfg.use_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project="behavioral-paired-tokens", name=cfg.run_name,
                   config=cfg.__dict__)

    best_eval = float("inf")
    step = 0
    model.train()
    for epoch in range(cfg.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        for i, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"], output_hidden_states=True)
            loss, metrics = compute_losses(out, batch, cfg.lambda_c)
            (loss / cfg.grad_accum).backward()

            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % cfg.log_steps == 0:
                    for k, v in metrics.items():
                        if not math.isnan(v):
                            writer.add_scalar(f"train/{k}", v, step)
                    if wandb:
                        wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
                    pbar.set_postfix(loss=f"{metrics['loss_total']:.4f}",
                                     var=f"{metrics['compress_hidden_variance']:.2e}")
                    if (cfg.lambda_c > 0
                            and metrics["compress_hidden_variance"] < COLLAPSE_VARIANCE_THRESHOLD):
                        print(f"\n[WARN] [COMPRESS] hidden-state variance "
                              f"{metrics['compress_hidden_variance']:.2e} below "
                              f"{COLLAPSE_VARIANCE_THRESHOLD} — possible collapse, "
                              f"lower lambda_c.")

                if step % cfg.eval_steps == 0:
                    eval_metrics = evaluate(model, eval_loader, device, cfg.lambda_c)
                    for k, v in eval_metrics.items():
                        writer.add_scalar(k, v, step)
                    if wandb:
                        wandb.log(eval_metrics, step=step)
                    if eval_metrics.get("eval/loss_ce", float("inf")) < best_eval:
                        best_eval = eval_metrics["eval/loss_ce"]
                        save_checkpoint(model, tokenizer, cfg, "best")

                if step % cfg.save_steps == 0:
                    save_checkpoint(model, tokenizer, cfg, "last")

    # final eval so short runs (debug) still produce a best checkpoint
    eval_metrics = evaluate(model, eval_loader, device, cfg.lambda_c)
    for k, v in eval_metrics.items():
        writer.add_scalar(k, v, step)
    if eval_metrics.get("eval/loss_ce", float("inf")) < best_eval:
        best_eval = eval_metrics["eval/loss_ce"]
        save_checkpoint(model, tokenizer, cfg, "best")

    save_checkpoint(model, tokenizer, cfg, "last")
    writer.close()
    if wandb:
        wandb.finish()
    print(f"Done. Best eval loss_ce: {best_eval:.4f}. Checkpoints in {run_dir}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, help="load a saved config.json")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--run-name", type=str)
    p.add_argument("--model-name", type=str)
    p.add_argument("--train-file", type=str)
    p.add_argument("--eval-file", type=str)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--lambda-c", type=float, dest="lambda_c")
    p.add_argument("--use-wandb", action="store_true")
    args = p.parse_args()

    cfg = TrainConfig.load(args.config) if args.config else TrainConfig()
    for key in ("run_name", "model_name", "train_file", "eval_file",
                "epochs", "batch_size", "lr", "lambda_c"):
        val = getattr(args, key)
        if val is not None:
            setattr(cfg, key, val)
    if args.use_wandb:
        cfg.use_wandb = True
    if args.debug:
        cfg.debug = True
        cfg.apply_debug()
    return cfg


if __name__ == "__main__":
    train(parse_args())
