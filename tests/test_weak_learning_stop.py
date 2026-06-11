import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_diffusion import WeakLearningStopper
from scripts.train_diffusion import build_train_valid_datasets


def test_weak_learning_stop_triggers_on_core_plateau_and_baseline_gap():
    stopper = WeakLearningStopper(
        {
            "enabled": True,
            "monitor": "valid_loss",
            "baseline_valid_loss": 1.51667,
            "max_relative_gap": 0.20,
            "core_monitor": "valid_pos_loss",
            "core_min_delta": 0.005,
            "core_max": 0.985,
            "patience": 2,
            "min_validations": 6,
            "start_after": 14000,
            "inactive_metrics": ["valid_hard_negative_loss", "valid_copy_gate_loss"],
        }
    )
    rows = [
        {"step": 2000, "valid_loss": 2.1964, "valid_pos_loss": 1.0058},
        {"step": 4000, "valid_loss": 2.0277, "valid_pos_loss": 0.9981},
        {"step": 6000, "valid_loss": 1.9834, "valid_pos_loss": 1.0018},
        {"step": 8000, "valid_loss": 2.0034, "valid_pos_loss": 1.0032},
        {"step": 10000, "valid_loss": 1.9852, "valid_pos_loss": 0.9902},
        {"step": 12000, "valid_loss": 1.9456, "valid_pos_loss": 0.9958},
        {"step": 14000, "valid_loss": 1.9070, "valid_pos_loss": 1.0045},
        {"step": 16000, "valid_loss": 1.9357, "valid_pos_loss": 0.9941},
    ]
    updates = []
    for row in rows:
        row = {
            **row,
            "valid_hard_negative_loss": 0.0,
            "valid_copy_gate_loss": 0.0,
        }
        updates.append(stopper.update(row))

    assert not updates[-2].get("weak_stop_triggered")
    assert updates[-1]["weak_stop_triggered"] is True
    assert updates[-1]["weak_stop_baseline_gap"] is True
    assert updates[-1]["weak_stop_bad_core_validations"] == 2
    assert "valid_hard_negative_loss" in updates[-1]["weak_stop_inactive_metrics"]


def test_train_valid_datasets_use_explicit_heldout_manifest(tmp_path):
    train_record = tmp_path / "train.pt"
    val_record = tmp_path / "val.pt"
    train_record.write_bytes(b"placeholder")
    val_record.write_bytes(b"placeholder")
    train_manifest = tmp_path / "train_manifest.txt"
    val_manifest = tmp_path / "val_manifest.txt"
    train_manifest.write_text(f"{train_record.name}\n")
    val_manifest.write_text(f"{val_record.name}\n")

    train_set, valid_set = build_train_valid_datasets(
        {
            "data": {
                "kind": "pt_manifest",
                "manifest": str(train_manifest),
                "heldout_manifest": str(val_manifest),
            },
            "seed": 2026,
        },
        {"valid_fraction": 0.5},
    )

    assert train_set.manifest_path == train_manifest
    assert valid_set is not None
    assert valid_set.manifest_path == val_manifest
