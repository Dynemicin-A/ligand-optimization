# v3 Experiment Plan

## Current Goal

Make backbone v3 learn useful protein-conditioned H2L transformations, then
validate by sampling and molecule-level evaluation. Validation loss alone is not
enough; the model must produce chemically valid molecules that improve
target/source-relevant metrics.

## Active Runs

Use only v3 configs and the unified wrapper:

```bash
python scripts/ligopt_v3.py train-pretrain -- --device cuda
python scripts/ligopt_v3.py train-h2l -- --device cuda --init-model outputs/v3_pretrain/checkpoint_best.pt
```

## Stop Criteria

Stop and review instead of extending weak runs when:

- best validation loss does not beat the current baseline after the configured
  early window
- three validation rounds fail to improve by the configured `min_delta`
- atom-type loss rises while coordinate loss dominates training
- hard-negative and ranking signals remain inactive
- samples are invalid or collapse to source-like molecules

## Review Contract

Every completed, failed, early-stopped, or manually stopped run must produce:

- `quality_review.json`
- `quality_review.md`
- `improvement_review.json`
- `improvement_review.md`

Command:

```bash
python scripts/ligopt_v3.py review -- \
  --run-dir outputs/<run> \
  --config outputs/<run>/config.yaml \
  --log logs/<run>.log \
  --baseline-name current_best \
  --baseline-valid-loss 1.5313
```

## Next Design Levers

Prefer architectural or objective changes over generic denoising sweeps:

- strengthen pair trunk depth/edge update where validation supports it
- make hard-negative sampling/counts effective
- add or tune copy/mutate supervision
- add contact/clash/ranking auxiliary losses
- inspect generated molecules before judging a checkpoint as useful
