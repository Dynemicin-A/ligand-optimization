# Protein-Conditioned Full-Molecule Optimization

Diffusion-first backbone and executable pipeline for protein-conditioned hit-to-lead full-molecule optimization.

The project target is not fixed-scaffold completion. The first model treats the low-activity hit as a soft source condition and denoises a target ligand under a fixed protein pocket, allowing scaffold migration. Flow matching, BFN, and autoregressive variants are reserved for later ablations.

## What Runs Now

```bash
python3 tests/test_backbone.py
python3 -m pytest tests

python3 scripts/train_diffusion.py \
  --config configs/train_synthetic_tiny.yaml \
  --outdir outputs/diffusion_tiny \
  --device cpu \
  --max-steps 10

python3 scripts/sample_diffusion.py \
  --checkpoint outputs/diffusion_tiny/checkpoint_last.pt \
  --out outputs/diffusion_tiny/sample.pt \
  --sdf-out outputs/diffusion_tiny/sample.sdf \
  --device cpu \
  --num-steps 16
```

## Main Pipeline

1. Preprocess H2L triples: protein pocket, low-activity source hit, high-activity target lead.
2. Expand H2L pairs within each activity series, attach same-pocket hard negatives, and apply geometry augmentation.
3. Pretrain on broad protein-ligand complexes, then finetune on expanded H2L pairs.
4. Train diffusion denoiser with validation-based early stopping, EMA, hard-negative ranking, and best-checkpoint selection.
5. Sample optimized full molecules.
6. Export tensor/SDF artifacts.
7. Evaluate chemical quality, active rediscovery, structure-based quality, and scaffold migration.

See [PIPELINE.md](</Users/z/Desktop/ligand- optimization/PIPELINE.md>) for the paper-grade pipeline and [RUNNING.md](</Users/z/Desktop/ligand- optimization/RUNNING.md>) for commands.
