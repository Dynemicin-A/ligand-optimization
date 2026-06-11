#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from time import time

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pco_backbone.data import PTRecordDataset, collate_complex_records, move_batch_to_device  # noqa: E402
from train_diffusion import build_model, require_v3_config, split_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a v3 checkpoint on a manifest validation split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--weights", choices=["auto", "model", "ema"], default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum eval batches. Omit for config train.valid_batches; use 0 for full split.",
    )
    parser.add_argument("--baseline-name", default=None)
    parser.add_argument("--baseline-valid-loss", type=float, default=None)
    parser.add_argument("--sota-name", default=None)
    parser.add_argument("--sota-valid-loss", type=float, default=None)
    parser.add_argument(
        "--note",
        default="",
        help="Free-form note stored in the report, for example why external SOTA is not directly comparable.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def choose_checkpoint_state(ckpt: dict, weights: str) -> tuple[str, dict[str, torch.Tensor]]:
    use_ema = weights in {"auto", "ema"} and "ema_model_state" in ckpt
    if weights == "ema" and not use_ema:
        raise KeyError("checkpoint has no ema_model_state")
    return ("ema" if use_ema else "model"), (ckpt["ema_model_state"] if use_ema else ckpt["model_state"])


def scalar_dict(out: dict[str, torch.Tensor]) -> dict[str, float]:
    scalar_metrics = {
        "positive_score",
        "negative_score",
        "score_gap",
        "hard_negative_count",
        "copy_gate_accuracy",
        "distogram_accuracy",
        "contact_accuracy",
        "ranking_accuracy",
    }
    metrics: dict[str, float] = {}
    for key, value in out.items():
        if value.dim() == 0 and (key == "loss" or key.endswith("loss") or key in scalar_metrics):
            metrics[key] = float(value.detach().cpu())
    return metrics


@torch.no_grad()
def evaluate(model, loader, device: torch.device, max_batches: int | None) -> tuple[dict[str, float], int, int]:
    model.eval()
    sums: dict[str, float] = {}
    total_records = 0
    n_batches = 0
    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        batch_size = int(batch["ligand_batch"].max().item() + 1) if batch["ligand_batch"].numel() else 0
        batch = move_batch_to_device(batch, device)
        metrics = scalar_dict(model.training_loss(batch))
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
        total_records += batch_size
        n_batches += 1
    averaged = {key: value / max(n_batches, 1) for key, value in sums.items()}
    return averaged, n_batches, total_records


def comparison(value: float | None, target: float | None) -> dict[str, float | None]:
    if value is None or target is None:
        return {"delta": None, "relative_delta": None}
    delta = value - target
    return {"delta": delta, "relative_delta": delta / target if target else None}


def write_markdown(path: Path, payload: dict) -> None:
    eval_metrics = payload["eval_metrics"]
    lines = [
        "# Checkpoint Loss Evaluation",
        "",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Config: `{payload['config']}`",
        f"- Weights: `{payload['weights']}`",
        f"- Device: `{payload['device']}`",
        f"- Split: `{payload['split']}`",
        f"- Batches: `{payload['n_batches']}`",
        f"- Records: `{payload['n_records']}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in sorted(eval_metrics):
        lines.append(f"| `{key}` | {eval_metrics[key]:.6f} |")
    lines.extend(["", "## Comparisons", "", "| reference | valid_loss | delta | relative |", "|---|---:|---:|---:|"])
    for item in payload["comparisons"]:
        valid_loss = item.get("valid_loss")
        delta = item.get("delta")
        rel = item.get("relative_delta")
        valid_text = f"{valid_loss:.6f}" if isinstance(valid_loss, float) and math.isfinite(valid_loss) else "n/a"
        delta_text = f"{delta:+.6f}" if isinstance(delta, float) and math.isfinite(delta) else "n/a"
        rel_text = f"{rel:+.2%}" if isinstance(rel, float) and math.isfinite(rel) else "n/a"
        lines.append(f"| {item['name']} | {valid_text} | {delta_text} | {rel_text} |")
    note = payload.get("note")
    if note:
        lines.extend(["", "## Note", "", note])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    require_v3_config(cfg, args.config)
    train_cfg = cfg.get("train", {}) or {}
    device = resolve_device(args.device)

    dataset = PTRecordDataset(cfg["data"]["manifest"])
    _train_set, valid_set = split_dataset(dataset, float(train_cfg.get("valid_fraction", 0.0)), int(cfg.get("seed", 2024)))
    split_name = "valid_fraction"
    if valid_set is None and cfg.get("data", {}).get("heldout_manifest"):
        valid_set = PTRecordDataset(cfg["data"]["heldout_manifest"])
        split_name = "heldout_manifest"
    if valid_set is None:
        raise ValueError("no validation split is available; set train.valid_fraction or data.heldout_manifest")

    batch_size = args.batch_size or int(train_cfg.get("batch_size", 4))
    max_batches = args.max_batches
    if max_batches is None:
        max_batches = int(train_cfg.get("valid_batches", 0) or 0)
    if max_batches == 0:
        max_batches = None
    loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_complex_records,
        pin_memory=False,
        drop_last=False,
    )

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights_name, state = choose_checkpoint_state(ckpt, args.weights)
    model.load_state_dict(state)
    started = time()
    metrics, n_batches, n_records = evaluate(model, loader, device, max_batches)
    elapsed_sec = time() - started
    valid_loss = metrics.get("loss")
    comparisons = []
    for name, target in [(args.baseline_name, args.baseline_valid_loss), (args.sota_name, args.sota_valid_loss)]:
        if not name:
            continue
        item = {"name": name, "valid_loss": target}
        item.update(comparison(valid_loss, target))
        comparisons.append(item)
    payload = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "weights": weights_name,
        "device": str(device),
        "split": split_name,
        "max_batches": max_batches,
        "n_batches": n_batches,
        "n_records": n_records,
        "elapsed_sec": elapsed_sec,
        "eval_metrics": metrics,
        "valid_loss": valid_loss,
        "comparisons": comparisons,
        "baseline": {"name": args.baseline_name, "valid_loss": args.baseline_valid_loss},
        "sota": {"name": args.sota_name, "valid_loss": args.sota_valid_loss},
        "checkpoint_step": int(ckpt.get("step", -1)),
        "backbone_config": ckpt.get("backbone_config"),
        "diffusion_config": ckpt.get("diffusion_config"),
        "note": args.note,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(args.out.with_suffix(".md"), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
