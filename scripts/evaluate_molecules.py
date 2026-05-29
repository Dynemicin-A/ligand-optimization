#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pco_backbone.metrics import (  # noqa: E402
    compute_molecule_metrics,
    load_mols_from_sdf,
    load_smiles,
    mols_from_smiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated molecules.")
    gen = parser.add_mutually_exclusive_group(required=True)
    gen.add_argument("--generated-sdf", type=Path)
    gen.add_argument("--generated-smiles", type=Path)
    parser.add_argument("--reference-smiles", type=Path, default=None)
    parser.add_argument("--source-smiles", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/eval_metrics.json")
    parser.add_argument("--rediscovery-threshold", type=float, default=0.7)
    parser.add_argument(
        "--expected-total",
        type=int,
        default=None,
        help="Pad missing/failed generations with None so validity is computed against the attempted sample count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generated_sdf is not None:
        generated = load_mols_from_sdf(args.generated_sdf)
    else:
        generated = mols_from_smiles(load_smiles(args.generated_smiles))
    if args.expected_total is not None and args.expected_total > len(generated):
        generated.extend([None] * (args.expected_total - len(generated)))
    references = mols_from_smiles(load_smiles(args.reference_smiles)) if args.reference_smiles else None
    sources = mols_from_smiles(load_smiles(args.source_smiles)) if args.source_smiles else None
    metrics = compute_molecule_metrics(
        generated,
        reference_mols=references,
        source_mols=sources,
        rediscovery_similarity_threshold=args.rediscovery_threshold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(metrics)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
