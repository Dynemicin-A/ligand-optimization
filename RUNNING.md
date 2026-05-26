# Running the Diffusion Backbone

## Smoke Test

```bash
python3 tests/test_backbone.py
```

## Train

```bash
python3 scripts/train_diffusion.py \
  --config configs/train_synthetic_tiny.yaml \
  --outdir outputs/diffusion_tiny \
  --device cpu
```

This default config trains on deterministic synthetic H2L-like records. It is meant to verify the whole program path: dataset, collate, model, diffusion loss, optimizer, metrics, and checkpoint writing.

Outputs:

- `outputs/diffusion_tiny/config.yaml`
- `outputs/diffusion_tiny/metrics.jsonl`
- `outputs/diffusion_tiny/checkpoint_last.pt`

## Sample

```bash
python3 scripts/sample_diffusion.py \
  --checkpoint outputs/diffusion_tiny/checkpoint_last.pt \
  --out outputs/diffusion_tiny/sample.pt \
  --device cpu \
  --num-steps 32 \
  --num-ligand-atoms 12
```

The sampler currently writes a tensor `.pt` artifact, not an SDF. Molecule reconstruction is a later step.

## Real Data Manifest

To train on preprocessed H2L records, switch the config data section:

```yaml
data:
  kind: pt_manifest
  manifest: /path/to/train_manifest.txt
```

The manifest contains one `.pt` record path per line. Each record must include:

- `protein_atom_type`
- `protein_pos`
- `source_atom_type`
- `source_pos`
- `ligand_atom_type`
- `ligand_pos`

Optional bond labels:

- `ligand_bond_edge_index`
- `ligand_bond_type`
