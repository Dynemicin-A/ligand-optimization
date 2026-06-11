# Running v3

## Tests

```bash
python -m pytest tests
```

## Unified CLI

All public workflows use:

```bash
python scripts/ligopt_v3.py <command> -- <args-for-underlying-script>
```

Commands:

- `preprocess`
- `preprocess-molgenbench-h2l`
- `train-pretrain`
- `train-h2l`
- `sample-h2l`
- `review`

## Preprocess Examples

```bash
python scripts/ligopt_v3.py preprocess -- \
  --csv data/sbdd/train_complexes.csv \
  --outdir data/pretrain_complexes/train \
  --source-mode self \
  --num-workers 16

python scripts/ligopt_v3.py preprocess-molgenbench-h2l -- \
  --root data/raw/MolGenBench_Version3 \
  --outdir data/processed_h2l/train \
  --pair-mode all_ordered \
  --hard-negative-mode within_series
```

## Training

```bash
python scripts/ligopt_v3.py train-pretrain -- --device cuda

python scripts/ligopt_v3.py train-h2l -- \
  --device cuda \
  --init-model outputs/v3_pretrain/checkpoint_best.pt \
  --init-weights auto
```

Default outputs:

- `outputs/v3_pretrain`
- `outputs/v3_h2l`

You can override with `--outdir` after the separator:

```bash
python scripts/ligopt_v3.py train-h2l -- --outdir outputs/my_v3_h2l_run --device cuda
```

## Sampling

```bash
python scripts/ligopt_v3.py sample-h2l -- \
  --checkpoint outputs/v3_h2l/checkpoint_best.pt \
  --manifest data/processed_h2l/val/manifest.txt \
  --outdir outputs/v3_h2l/generated \
  --num-samples 256 \
  --num-steps 64
```

## Review

```bash
python scripts/ligopt_v3.py review -- \
  --run-dir outputs/<run> \
  --config outputs/<run>/config.yaml \
  --log logs/<run>.log \
  --baseline-name current_best \
  --baseline-valid-loss 1.5313
```

Required review artifacts:

- `quality_review.json`
- `quality_review.md`
- `improvement_review.json`
- `improvement_review.md`

## Slurm

The retained Slurm jobs call the same wrapper:

- `jobs/preprocess_complex_dataset.slurm`
- `jobs/preprocess_molgenbench_h2l.slurm`
- `jobs/sample_eval_h2l_after_train.slurm`

No pre-v3 train Slurm jobs are kept.
