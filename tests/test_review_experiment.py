import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_review_experiment_writes_quality_and_improvement_reviews(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("a.pt\nb.pt\n")
    config = tmp_path / "config.yaml"
    config.write_text(
        "data:\n"
        "  manifest: manifest.txt\n"
        "train:\n"
        "  early_stopping:\n"
        "    enabled: true\n"
    )
    log = tmp_path / "train.log"
    log.write_text("done: step=30000, checkpoint=checkpoint_last.pt\n")

    metrics = [
        {"step": 100, "train_loss": 2.0, "train_pos_loss": 1.1, "train_atom_loss": 0.7, "train_bond_loss": 0.2},
        {
            "step": 10000,
            "valid_loss": 1.50,
            "valid_pos_loss": 0.98,
            "valid_atom_loss": 0.49,
            "valid_bond_loss": 0.18,
        },
        {"step": 20000, "train_loss": 1.2, "train_pos_loss": 0.9, "train_atom_loss": 0.25, "train_bond_loss": 0.1},
        {
            "step": 30000,
            "valid_loss": 1.57,
            "valid_pos_loss": 1.02,
            "valid_atom_loss": 0.51,
            "valid_bond_loss": 0.19,
        },
    ]
    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in metrics) + "\n")
    (run_dir / "checkpoint_step_10000.pt").write_text("ckpt")
    (run_dir / "checkpoint_last.pt").write_text("ckpt")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "review_experiment.py"),
            "--run-dir",
            str(run_dir),
            "--config",
            str(config),
            "--log",
            str(log),
            "--baseline-name",
            "baseline",
            "--baseline-valid-loss",
            "1.53",
            "--overfit-delta",
            "0.04",
            "--overfit-steps",
            "15000",
        ],
        cwd=tmp_path,
        check=True,
    )

    quality = json.loads((run_dir / "quality_review.json").read_text())
    improvement = json.loads((run_dir / "improvement_review.json").read_text())

    assert quality["metrics"]["best_valid"]["step"] == 10000
    assert quality["metrics"]["saved_best"]["step"] == 10000
    assert quality["metrics"]["overfit_like_regression"] is True
    assert quality["manifest_count"] == 2
    assert any("validation regression" in warning for warning in quality["warnings"])
    assert any("validation has regressed" in item for item in improvement["diagnoses"])
    assert (run_dir / "quality_review.md").exists()
    assert (run_dir / "improvement_review.md").exists()
