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

## Expanded H2L Pairs And Augmentation

The MolGenBench H2L preprocessor can expand each target series from only
reference-hit-to-lead pairs into all lower-activity to higher-activity pairs
when affinity metadata is available:

```bash
python3 scripts/preprocess_molgenbench_h2l.py \
  --root data/raw/molgenbench_v3/extracted/MolGenBench_Version3 \
  --outdir data/processed_h2l_expanded/train \
  --pair-mode all_ordered \
  --augment-copies 4 \
  --global-random-rotate \
  --global-translate-sigma 0.5 \
  --ligand-noise-sigma 0.05 \
  --source-noise-sigma 0.03
```

On Slurm, use the existing job with env overrides:

```bash
PAIR_MODE=all_ordered \
AUGMENT_COPIES=4 \
GLOBAL_RANDOM_ROTATE=1 \
GLOBAL_TRANSLATE_SIGMA=0.5 \
LIGAND_NOISE_SIGMA=0.05 \
SOURCE_NOISE_SIGMA=0.03 \
OUTDIR=/home/scc/pb22000262/ligand-optimization/data/processed_h2l_expanded/train \
sbatch jobs/preprocess_molgenbench_h2l.slurm
```

The regularized A100 H2L config now uses a smaller backbone, dropout, higher
weight decay, validation every 1000 steps, and early stopping. It writes
`checkpoint_best.pt` whenever `valid_loss` improves.

## Pretrain Then Finetune

Prepare a pretraining CSV from PDBbind, CrossDocked, Binding MOAD, or another
protein-ligand complex source:

```csv
record_id,protein_path,ligand_path
1abc_0,/path/1abc_pocket.pdb,/path/1abc_ligand.sdf
```

Preprocess and train the protein-ligand denoising pretrain stage:

```bash
CSV=/path/to/pretrain_complexes.csv \
OUTDIR=/home/scc/pb22000262/ligand-optimization/data/pretrain_complexes/train \
sbatch jobs/preprocess_complex_pretrain.slurm

sbatch jobs/train_pretrain_complexes_a100_1gpu_4h.slurm
```

Finetune H2L from the pretraining best checkpoint:

```bash
INIT_MODEL=/home/scc/pb22000262/ligand-optimization/outputs/pretrain_run/checkpoint_best.pt \
OUTDIR=/home/scc/pb22000262/ligand-optimization/outputs/h2l_finetune_expanded \
CONFIG=configs/train_h2l_a100_1gpu_4h.yaml \
sbatch jobs/train_h2l_a100_1gpu_4h.slurm
```

Use `checkpoint_best.pt` for final sampling/evaluation rather than the last
checkpoint if validation has plateaued or regressed.

## Preprocess Real H2L Triples

```bash
python3 scripts/preprocess_h2l.py \
  --csv data/h2l/train.csv \
  --outdir data/processed_h2l/train \
  --pocket-radius 10
```

The CSV must contain:

- `protein_path`
- `source_ligand_path`
- `target_ligand_path`

## Evaluate Molecules

```bash
python3 scripts/evaluate_molecules.py \
  --generated-sdf outputs/diffusion_tiny/sample.sdf \
  --reference-smiles data/h2l/reference_actives.smi \
  --source-smiles data/h2l/source_hits.smi \
  --out outputs/diffusion_tiny/eval_metrics.json
```
