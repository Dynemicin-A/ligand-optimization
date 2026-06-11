# Current Work Summary

## Objective

The project goal is a protein-conditioned full-molecule H2L optimizer. The
active implementation is backbone v3 only. The model should use a low-activity
source hit as soft context, generate an optimized ligand under a protein pocket,
and support scaffold migration when useful.

## What Was Established

- Raw SBDD download/resume management exists for PDBbind, Binding MOAD, and
  CrossDocked.
- PDBbind preprocessing exposed bad records where protein-like files were used
  as ligands; the clean manifest path is
  `data/pretrain_pdbbind_v2020/train/manifest_clean_lig256_prot1200.txt`.
- ChEMBL/H2L low-to-high pairs and processed train/val manifests were prepared
  for H2L finetuning.
- Weak experiment lines should be stopped early, reviewed, and redesigned.
  Repeated train/valid gap, rising atom loss, inactive hard-negative loss, and
  poor H2L validation are redesign signals.

## Current Code State

Backbone v3 is the only active model/config interface:

- `configs/pretrain_complexes_backbone_v3_4090.yaml`
- `configs/train_h2l_chembl_backbone_v3_4090.yaml`
- `scripts/ligopt_v3.py`
- `src/pco_backbone/model.py`
- `src/pco_backbone/layers.py`
- `src/pco_backbone/diffusion.py`
- `src/pco_backbone/records.py`

The v3 backbone includes:

- gaussian + cosine radial features
- cosine radial envelope
- layer norm
- residual FFN
- edge gates
- pair trunk
- distogram/contact auxiliary heads
- copy/mutate gate
- source-as-negative ranking signal

## Interface Decision

All public workflows now use:

```bash
python scripts/ligopt_v3.py <command> -- <args>
```

Training rejects non-v3 configs. Old non-v3 configs and old public
preprocessing/training/sampling entry points were removed.

## Unified Data Contract

Dataset-specific import scripts may handle CSV/JSONL/path-table variants, but
the saved `.pt` record must use canonical v3 keys:

- `protein_atom_type`, `protein_pos`
- `source_atom_type`, `source_pos`, `source_edge_index`, `source_bond_type`
- `ligand_atom_type`, `ligand_pos`, `ligand_bond_edge_index`, `ligand_bond_type`
- optional hard negative ligand keys

The training layer no longer performs legacy alias normalization.

## Next Steps

1. Verify the v3-only cleanup with full tests and py_compile.
2. Run the plain v3 baseline first: PDBbind pretraining on the clean manifest,
   then H2L finetuning from the best v3 pretrain checkpoint.
3. Keep MaskDiT-style structured masking and Differential-DiT-style
   differential attention as ablation lines only; do not mix them into the
   baseline until the plain v3 baseline has a complete pretrain + H2L result.
4. Stop runs that fail the strict criteria and run post-run review immediately.
5. If v3 H2L still underperforms, prioritize H2L-specific mechanism changes:
   hard-negative effectiveness, copy/mutate gate supervision, contact/clash
   losses, and ranking/evaluation alignment.
