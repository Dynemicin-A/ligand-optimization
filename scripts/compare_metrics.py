#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


CORE_WEIGHTS = {
    "pos_loss": 1.0,
    "atom_loss": 1.0,
    "bond_loss": 0.2,
}
AUX_WEIGHTS = {
    "hard_negative_loss": "hard_negative_loss_weight",
    "distogram_loss": "distogram_loss_weight",
    "contact_loss": "contact_loss_weight",
    "copy_gate_loss": "copy_gate_loss_weight",
}


def parse_run(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name, Path(path)
    path = Path(value)
    return path.name, path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    with config_path.open() as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def finite_float(value: Any) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def select_row(rows: list[dict[str, Any]], prefix: str, step: int | None) -> dict[str, Any] | None:
    key = f"{prefix}_loss"
    candidates = [row for row in rows if key in row]
    if not candidates:
        return None
    if step is None:
        return candidates[-1]
    return min(candidates, key=lambda row: abs(int(row.get("step", 0)) - step))


def metric(row: dict[str, Any], prefix: str, name: str) -> float:
    return finite_float(row.get(f"{prefix}_{name}")) or 0.0


def summarize_run(name: str, run_dir: Path, step: int | None, prefix: str) -> dict[str, Any]:
    rows = load_jsonl(run_dir / "metrics.jsonl")
    cfg = load_config(run_dir)
    diffusion = cfg.get("diffusion", {}) if isinstance(cfg.get("diffusion"), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
    row = select_row(rows, prefix, step)
    if row is None:
        return {"name": name, "run_dir": str(run_dir), "prefix": prefix, "missing": True}

    core_terms = {key: metric(row, prefix, key) for key in CORE_WEIGHTS}
    weighted_core = sum(CORE_WEIGHTS[key] * value for key, value in core_terms.items())
    aux_terms = {key: metric(row, prefix, key) for key in AUX_WEIGHTS}
    aux_weights = {
        key: float(diffusion.get(weight_key, 0.0) or 0.0)
        for key, weight_key in AUX_WEIGHTS.items()
    }
    weighted_aux = sum(aux_weights[key] * aux_terms[key] for key in AUX_WEIGHTS)
    reported_loss = finite_float(row.get(f"{prefix}_loss"))
    return {
        "name": name,
        "run_dir": str(run_dir),
        "prefix": prefix,
        "step": int(row.get("step", 0)),
        "reported_loss": reported_loss,
        "weighted_core_loss": weighted_core,
        "weighted_aux_loss": weighted_aux,
        "weighted_reconstructed_loss": weighted_core + weighted_aux,
        "core_terms": core_terms,
        "aux_terms": aux_terms,
        "aux_weights": aux_weights,
        "num_blocks": model_cfg.get("num_blocks"),
        "pair_num_blocks": model_cfg.get("pair_num_blocks"),
        "batch_size": train_cfg.get("batch_size"),
        "lr": train_cfg.get("lr"),
        "score_gap": finite_float(row.get(f"{prefix}_score_gap")),
        "hard_negative_count": finite_float(row.get(f"{prefix}_hard_negative_count")),
    }


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = ["# Metrics Comparison", ""]
    for item in summaries:
        if item.get("missing"):
            lines.append(f"- `{item['name']}`: missing `{item['prefix']}_loss` row")
            continue
        lines.extend(
            [
                f"## {item['name']}",
                f"- Run: `{item['run_dir']}`",
                f"- Step: `{item['step']}`",
                f"- Reported loss: `{item['reported_loss']}`",
                f"- Weighted core loss: `{item['weighted_core_loss']:.6f}`",
                f"- Weighted aux loss: `{item['weighted_aux_loss']:.6f}`",
                f"- Core terms: `{item['core_terms']}`",
                f"- Aux terms: `{item['aux_terms']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare training metrics with core/aux loss decomposition.")
    parser.add_argument("--run", action="append", required=True, help="Run directory, optionally NAME=PATH.")
    parser.add_argument("--step", type=int, default=None, help="Select closest metric row to this step.")
    parser.add_argument("--prefix", choices=["valid", "train"], default="valid")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [
        summarize_run(name, path, args.step, args.prefix)
        for name, path in (parse_run(value) for value in args.run)
    ]
    payload = {"prefix": args.prefix, "requested_step": args.step, "runs": summaries}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end="")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.markdown, summaries)


if __name__ == "__main__":
    main()
