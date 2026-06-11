# Backbone v3

## Goal

Backbone v3 is the only active architecture line. It targets
protein-conditioned full-molecule hit-to-lead optimization:

```text
input:
  protein pocket P
  low-activity source hit M_low

output:
  optimized molecule M_gen
```

The source hit is a condition, not a hard scaffold mask.

## Structure

```text
protein atoms + coordinates
source-hit graph + coordinates
noisy target ligand graph + coordinates + time embedding

single/pair trunk:
  local ligand, source, and protein messages
  protein-ligand and source-ligand cross messages
  pair feature updates on sparse edges
  cosine-enveloped gaussian radial features
  pre-norm residual FFN blocks
  edge-gated messages
  equivariant ligand coordinate update

heads:
  coordinate noise
  atom type logits
  bond type logits
  protein-ligand distogram
  protein-ligand contact
  source-to-target copy/mutate gate
  source-as-negative ranking score
```

## Why This Shape

- Pair trunk: follows the module organization used by stronger biomolecular
  diffusion/structure backbones instead of only deepening atom MLPs.
- Cosine radial envelope: avoids hard KNN/cutoff discontinuities.
- Residual FFN + layer norm: stabilizes deeper message passing.
- Edge gates: lets the model suppress noisy ligand/source/protein edges.
- Distogram/contact auxiliary losses: force pocket interaction learning instead
  of only denoising coordinates.
- Copy/mutate gate: gives H2L a direct mechanism for deciding whether to retain,
  move, insert, or delete source-hit structure.

See `BACKBONE_V3_REFERENCES.md` for the reference backbone rationale.

## Implementation Files

- `src/pco_backbone/model.py`
- `src/pco_backbone/layers.py`
- `src/pco_backbone/diffusion.py`
- `src/pco_backbone/data.py`
- `src/pco_backbone/records.py`
- `scripts/train_diffusion.py`
- `scripts/ligopt_v3.py`
- `configs/pretrain_complexes_backbone_v3_4090.yaml`
- `configs/train_h2l_chembl_backbone_v3_4090.yaml`

## Record Contract

All datasets enter the model through `src/pco_backbone/records.py`.
Dataset-specific naming is handled before record construction. Training records
must already use canonical v3 keys; legacy aliases are intentionally removed.

Required keys:

- `protein_atom_type`, `protein_pos`
- `source_atom_type`, `source_pos`
- `ligand_atom_type`, `ligand_pos`

Optional keys:

- `source_edge_index`, `source_bond_type`
- `ligand_bond_edge_index`, `ligand_bond_type`
- `negative_ligand_atom_type`, `negative_ligand_pos`
- `negative_ligand_bond_edge_index`, `negative_ligand_bond_type`

## Run

```bash
python scripts/ligopt_v3.py train-pretrain -- --device cuda
python scripts/ligopt_v3.py train-h2l -- --device cuda --init-model outputs/v3_pretrain/checkpoint_best.pt
```

Every completed, failed, or manually stopped run must be reviewed before its GPU
is reused:

```bash
python scripts/ligopt_v3.py review -- --run-dir outputs/<run> --config outputs/<run>/config.yaml --log logs/<run>.log --baseline-name current_best --baseline-valid-loss 1.5313
```
