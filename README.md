# Protein-Conditioned Ligand Optimization

This repo now exposes one active model line: **backbone v3** for
protein-conditioned full-molecule hit-to-lead optimization.

The task is not fixed-scaffold completion. The source hit is a soft condition;
the model may copy, mutate, or migrate scaffold under a fixed protein pocket.

## Active Interface

Use the unified wrapper for all runnable workflows:

```bash
python scripts/ligopt_v3.py preprocess -- --csv data/complexes.csv --outdir data/pretrain/train
python scripts/ligopt_v3.py preprocess-molgenbench-h2l -- --root data/raw/MolGenBench_Version3 --outdir data/h2l/train
python scripts/ligopt_v3.py train-pretrain -- --device cuda
python scripts/ligopt_v3.py train-h2l -- --device cuda --init-model outputs/v3_pretrain/checkpoint_best.pt
python scripts/ligopt_v3.py sample-h2l -- --checkpoint outputs/v3_h2l/checkpoint_best.pt --manifest data/h2l/val/manifest.txt --outdir outputs/v3_h2l/samples
python scripts/ligopt_v3.py review -- --run-dir outputs/v3_h2l --config outputs/v3_h2l/config.yaml --log logs/v3_h2l.log
```

The low-level scripts remain implementation modules, but the public entry point
is `scripts/ligopt_v3.py`.

## Active Configs

- `configs/pretrain_complexes_backbone_v3_4090.yaml`
- `configs/train_h2l_chembl_backbone_v3_4090.yaml`

`scripts/train_diffusion.py` rejects non-v3 configs. Old non-v3 configs and
pre-v3 backbone configs have been removed.

## Unified Record Schema

Preprocessors may accept dataset-specific CSV columns, but saved `.pt` records
must use the canonical model keys:

- `protein_atom_type`, `protein_pos`
- `source_atom_type`, `source_pos`, `source_edge_index`, `source_bond_type`
- `ligand_atom_type`, `ligand_pos`, `ligand_bond_edge_index`, `ligand_bond_type`
- optional `negative_ligand_atom_type`, `negative_ligand_pos`,
  `negative_ligand_bond_edge_index`, `negative_ligand_bond_type`

The training/data layer no longer normalizes legacy aliases.

## Verification

```bash
python -m pytest tests
python -m py_compile scripts/ligopt_v3.py scripts/train_diffusion.py scripts/preprocess_complex_dataset.py scripts/preprocess_molgenbench_h2l.py scripts/sample_h2l_manifest.py
```

## Data Bootstrap

```bash
DATA_ROOT=/home/zhangxuanhao/zxh/datasets \
HF_ENDPOINT=https://hf-mirror.com \
bash scripts/manage_sbdd_dataset_downloads.sh
```

This manager resumes PDBbind v2020, Binding MOAD, and CrossDocked raw downloads.
