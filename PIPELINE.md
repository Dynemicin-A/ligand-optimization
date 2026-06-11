# v3 Pipeline

## Objective

Build a protein-conditioned full-molecule optimizer that turns a low-activity
hit into a better lead-like molecule while preserving pocket compatibility and
allowing scaffold migration.

## Data Flow

```text
raw complex / H2L files
  -> dataset-specific path table or directory discovery
  -> canonical v3 .pt records
  -> manifest.txt
  -> v3 pretrain or H2L finetune
  -> samples
  -> chemical + H2L evaluation
  -> mandatory quality/improvement review
```

## Preprocess

Generic SBDD complexes:

```bash
python scripts/ligopt_v3.py preprocess -- \
  --csv data/sbdd/train_complexes.csv \
  --outdir data/pretrain_complexes/train \
  --source-mode self \
  --num-workers 16
```

MolGenBench-style H2L:

```bash
python scripts/ligopt_v3.py preprocess-molgenbench-h2l -- \
  --root data/raw/MolGenBench_Version3 \
  --outdir data/processed_h2l/train \
  --pair-mode all_ordered \
  --hard-negative-mode within_series \
  --augment-copies 4
```

CSV/JSONL import may accept dataset column aliases, but the resulting `.pt`
records must be canonical v3 records.

## Train

Pretrain:

```bash
python scripts/ligopt_v3.py train-pretrain -- --device cuda
```

H2L finetune:

```bash
python scripts/ligopt_v3.py train-h2l -- \
  --device cuda \
  --init-model outputs/v3_pretrain/checkpoint_best.pt \
  --init-weights auto
```

The training script accepts only v3 configs with pair trunk, cosine radial
features, residual FFN, layer norm, edge gates, and `pt_manifest` data.

## Sample And Evaluate

```bash
python scripts/ligopt_v3.py sample-h2l -- \
  --checkpoint outputs/v3_h2l/checkpoint_best.pt \
  --manifest data/processed_h2l/val/manifest.txt \
  --outdir outputs/v3_h2l/generated \
  --num-samples 256 \
  --num-steps 64

python scripts/evaluate_molecules.py \
  --generated-sdf outputs/v3_h2l/generated/sdf \
  --reference-smiles outputs/v3_h2l/generated/reference.smi \
  --source-smiles outputs/v3_h2l/generated/source.smi \
  --out outputs/v3_h2l/eval_metrics.json
```

## Review Loop

Stop weak runs early. Signals include:

- validation loss fails to beat the current baseline after the configured start
  window
- validation loss regresses for repeated validations
- atom-type loss rises while train loss keeps dropping
- hard-negative count/loss stays ineffective
- sampling/evaluation shows poor validity or no source-to-target improvement

After any stop:

```bash
python scripts/ligopt_v3.py review -- \
  --run-dir outputs/<run> \
  --config outputs/<run>/config.yaml \
  --log logs/<run>.log \
  --baseline-name current_best \
  --baseline-valid-loss 1.5313
```

The next run should be justified by the review, not by generic hyperparameter
sweeping.
