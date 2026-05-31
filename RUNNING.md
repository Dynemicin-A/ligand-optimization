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

Optional hard-negative labels used by the ranking loss:

- `negative_ligand_atom_type`
- `negative_ligand_pos`
- `negative_ligand_bond_edge_index`
- `negative_ligand_bond_type`

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
  --source-noise-sigma 0.03 \
  --negative-noise-sigma 0.03 \
  --hard-negative-mode within_series \
  --hard-negative-min-target-similarity 0.2
```

On Slurm, use the existing job with env overrides:

```bash
PAIR_MODE=all_ordered \
AUGMENT_COPIES=4 \
GLOBAL_RANDOM_ROTATE=1 \
GLOBAL_TRANSLATE_SIGMA=0.5 \
LIGAND_NOISE_SIGMA=0.05 \
SOURCE_NOISE_SIGMA=0.03 \
NEGATIVE_NOISE_SIGMA=0.03 \
HARD_NEGATIVE_MODE=within_series \
HARD_NEGATIVE_MIN_TARGET_SIMILARITY=0.2 \
OUTDIR=/home/scc/pb22000262/ligand-optimization/data/processed_h2l_expanded/train \
sbatch jobs/preprocess_molgenbench_h2l.slurm
```

The regularized A100 H2L config now uses a smaller backbone, dropout, higher
weight decay, EMA validation/sampling, hard-negative ranking loss, validation
every 1000 steps, and early stopping. It writes `checkpoint_best.pt` whenever
`valid_loss` improves.

## Pretrain Then Finetune

MolGenBench is not the main training source. Use it mainly for held-out H2L
evaluation and small sanity checks. The bulk pretraining data should come from
large protein-ligand complex sources such as PDBbind, CrossDocked, Binding MOAD,
or internal docking/experimental complex tables.

Bootstrap the public sources into a named dataset root before preprocessing:

```bash
DATA_ROOT=/home/zhangxuanhao/zxh/datasets \
HF_ENDPOINT=https://hf-mirror.com \
NUM_DOWNLOADS=4 \
bash scripts/manage_sbdd_dataset_downloads.sh
```

The manager starts missing downloads in the background and reports status. It
uses `scripts/download_sbdd_datasets.py` underneath, with resumable Zenodo
downloads for PDBbind v2020 and prepared Binding MOAD, and Hugging Face download
for CrossDocked2020. On networks where `huggingface.co` is reset, keep
`HF_ENDPOINT=https://hf-mirror.com`. `NUM_DOWNLOADS=4` enables conservative
parallel archive downloads for large Zenodo records.

To run one source synchronously:

```bash
python scripts/download_sbdd_datasets.py \
  --root /home/zhangxuanhao/zxh/datasets \
  --datasets pdbbind_v2020 \
  --max-attempts 50
```

The general ingestion entry point is `scripts/preprocess_complex_dataset.py`.
It accepts explicit CSV/TSV/JSONL path tables, which is the preferred route for
large datasets because it avoids brittle filename assumptions:

```csv
record_id,protein_path,ligand_path,target_id,series_id
1abc_0,/path/1abc_pocket.pdb,/path/1abc_ligand.sdf,1abc,pdbbind
```

It also supports `source_ligand_path` and `negative_ligand_path` when the source
is already an H2L hit or a known decoy. Without `source_ligand_path`, use
`--source-mode self` for SBDD pretraining.

Examples:

```bash
python scripts/preprocess_complex_dataset.py \
  --csv /path/to/pretrain_complexes.csv \
  --outdir data/pretrain_complexes/train \
  --source-mode self \
  --num-workers 16 \
  --skip-existing

python scripts/preprocess_complex_dataset.py \
  --jsonl /path/to/crossdocked_split.jsonl \
  --outdir data/crossdocked/train \
  --source-mode self \
  --num-workers 16

python scripts/preprocess_complex_dataset.py \
  --root /path/to/pdbbind/refined-set \
  --preset pdbbind \
  --outdir data/pdbbind_refined/train \
  --source-mode self \
  --ligands-per-protein all \
  --num-workers 16
```

On Slurm:

```bash
CSV=/path/to/pretrain_complexes.csv \
OUTDIR=/home/scc/pb22000262/ligand-optimization/data/pretrain_complexes/train \
NUM_WORKERS=4 \
SOURCE_MODE=self \
sbatch jobs/preprocess_complex_dataset.slurm
```

Train the protein-ligand denoising pretrain stage:

```bash
sbatch jobs/train_pretrain_complexes_a100_1gpu_4h.slurm
```

Finetune H2L from the pretraining best checkpoint:

```bash
INIT_MODEL=/home/scc/pb22000262/ligand-optimization/outputs/pretrain_run/checkpoint_best.pt \
OUTDIR=/home/scc/pb22000262/ligand-optimization/outputs/h2l_finetune_expanded \
DATA_MANIFEST=/home/scc/pb22000262/ligand-optimization/data/processed_h2l_expanded/train/manifest.txt \
CONFIG=configs/train_h2l_a100_1gpu_4h.yaml \
sbatch jobs/train_h2l_a100_1gpu_4h.slurm
```

Use `checkpoint_best.pt` for final sampling/evaluation rather than the last
checkpoint if validation has plateaued or regressed. Sampling uses EMA weights
automatically when the checkpoint contains `ema_model_state`.

MolGenBench H2L should be treated as a held-out benchmark for the paper-grade
run. Do not use the MolGenBench H2L test split as the main training source;
pretrain on broad SBDD complexes and finetune on non-overlapping H2L pairs, then
point `MANIFEST` in `jobs/sample_eval_h2l_after_train.slurm` at the held-out
MolGenBench H2L manifest for evaluation only.

## 4090 Long-Run Node

For unmanaged 4090 machines, use the long configs and pin one process per GPU:

```bash
python scripts/split_manifest_by_target.py \
  --manifest data/processed_h2l_v4_expanded/train/manifest.txt \
  --outdir data/processed_h2l_v4_expanded/split \
  --heldout-fraction 0.2

CUDA_VISIBLE_DEVICES=3 python scripts/train_diffusion.py \
  --config configs/train_h2l_4090_long.yaml \
  --outdir outputs/h2l_v4_4090_hn_long

CUDA_VISIBLE_DEVICES=6 python scripts/train_diffusion.py \
  --config configs/train_h2l_4090_no_hn_long.yaml \
  --outdir outputs/h2l_v4_4090_no_hn_long
```

Both configs are deliberately budgeted for long runs (`max_steps=240000`) while
still using EMA validation, `checkpoint_best.pt`, and early stopping after a
minimum warmup.

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
