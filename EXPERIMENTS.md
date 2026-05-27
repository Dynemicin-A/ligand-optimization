# Experiment Plan

## Immediate Local Checks

1. Unit tests:
   ```bash
   python3 tests/test_backbone.py
   python3 -m pytest tests
   ```

2. Synthetic training:
   ```bash
   python3 scripts/train_diffusion.py --config configs/train_synthetic_tiny.yaml --outdir outputs/smoke --device cpu --max-steps 10
   ```

3. Synthetic sampling:
   ```bash
   python3 scripts/sample_diffusion.py --checkpoint outputs/smoke/checkpoint_last.pt --out outputs/smoke/sample.pt --sdf-out outputs/smoke/sample.sdf --device cpu --num-steps 16
   ```

4. Chemical evaluation:
   ```bash
   python3 scripts/evaluate_molecules.py --generated-sdf outputs/smoke/sample.sdf --out outputs/smoke/eval.json
   ```

## Main Paper Experiments

| Group | Data | Method | Primary metrics |
| --- | --- | --- | --- |
| A0 | H2L | fixed scaffold baseline | Validity, HitRediscover, Vina, scaffold retention |
| A1 | H2L | flexible scaffold baseline | Validity, HitRediscover, Vina, scaffold RMSD |
| B0 | H2L | diffusion with source regularization | Validity, HitRediscover, source similarity |
| B1 | H2L | diffusion with free scaffold migration | Validity, HitRediscover, Vina, scaffold hopping |

## Later Ablations

- Replace diffusion objective with flow matching.
- Replace source conditioning style: graph-level only vs source-to-ligand cross message.
- Add affinity/property guidance.
- Add variable ligand-size proposal.
- Add robust RDKit/OpenBabel reconstruction.
- Add PoseBusters/GenBench3D evaluation.
