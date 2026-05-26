# Model Backbone Draft

## Design target

The backbone is for protein-conditioned full-molecule optimization, not fixed-scaffold completion. The first implementation path is diffusion. Flow matching, BFN, and autoregressive variants should be treated as later ablation/model-family comparisons, not as constraints on the first version.

Inputs:

- fixed protein pocket atoms and 3D coordinates
- noisy target ligand atom types and 3D coordinates
- optional low-activity source-hit molecule as a conditioning graph
- graph batch ids
- time/noise level `t`

Outputs:

- ligand coordinate noise/update
- ligand atom-type logits
- ligand bond logits on local ligand edges
- optional complex-level score for affinity/property/reranking heads

## Backbone structure

```text
protein atoms -> protein scalar encoder
source-hit atoms -> source scalar encoder (optional)
noisy target ligand atoms -> ligand scalar encoder + time embedding

repeat N blocks:
  protein-protein local scalar message passing
  source-source local scalar message passing (optional)
  ligand-ligand local message passing
  protein-to-ligand cross message passing
  source-to-ligand soft cross message passing (optional)
  source graph-level conditioning into ligand state (optional)
  equivariant ligand coordinate update

heads:
  atom logits
  coordinate noise/update
  bond logits
  complex score
```

This mirrors the common pattern in TargetDiff/DiffSBDD-style diffusion models: encode pocket context, jointly denoise ligand geometry and discrete molecular state, and keep protein coordinates fixed.

## Why this backbone

- Diffusion-first: `pos_update` is trained as coordinate noise in the first wrapper.
- Ablation-ready: the trunk does not prevent later flow/BFN heads, but those should come after the diffusion baseline is working.
- Protein-conditioned: ligand updates use protein-to-ligand cross messages in every block.
- Hit-to-lead-aware: the low-activity source molecule can condition the denoiser through soft cross messages without being treated as a fixed scaffold.
- Scaffold-migration-friendly: the model has no hard scaffold mask in the trunk; scaffold preservation can be added as a training loss or sampling constraint, not as the default architecture.
- Lightweight preliminary: the prototype only depends on PyTorch, because this workspace does not currently have PyG/e3nn/torch_scatter installed.

## Current implementation

- [scripts/train_diffusion.py](</Users/z/Desktop/ligand- optimization/scripts/train_diffusion.py>)
- [scripts/sample_diffusion.py](</Users/z/Desktop/ligand- optimization/scripts/sample_diffusion.py>)
- [src/pco_backbone/data.py](</Users/z/Desktop/ligand- optimization/src/pco_backbone/data.py>)
- [src/pco_backbone/model.py](</Users/z/Desktop/ligand- optimization/src/pco_backbone/model.py>)
- [src/pco_backbone/layers.py](</Users/z/Desktop/ligand- optimization/src/pco_backbone/layers.py>)
- [src/pco_backbone/diffusion.py](</Users/z/Desktop/ligand- optimization/src/pco_backbone/diffusion.py>)
- [configs/train_synthetic_tiny.yaml](</Users/z/Desktop/ligand- optimization/configs/train_synthetic_tiny.yaml>)
- [configs/backbone_tiny.yaml](</Users/z/Desktop/ligand- optimization/configs/backbone_tiny.yaml>)
- [configs/diffusion_tiny.yaml](</Users/z/Desktop/ligand- optimization/configs/diffusion_tiny.yaml>)
- [tests/test_backbone.py](</Users/z/Desktop/ligand- optimization/tests/test_backbone.py>)

This is a diffusion-first prototype, not a production model. The current wrapper uses:

- Gaussian noising on ligand coordinates.
- simple mask corruption on target ligand atom types.
- optional source-hit graph and source-to-ligand conditioning.
- atom-type cross entropy.
- coordinate-noise MSE.
- local KNN bond/non-bond cross entropy.

The next steps are:

1. replace synthetic records with real H2L preprocessed `.pt` records;
2. add reverse-diffusion sampling code;
3. define final atom/bond vocabularies;
4. add evaluation adapters for MolGenBench H2L and structure-based docking metrics;
5. add flow matching/BFN variants as ablations after the diffusion baseline is stable.

## Run

```bash
python3 scripts/train_diffusion.py \
  --config configs/train_synthetic_tiny.yaml \
  --outdir outputs/diffusion_tiny \
  --device cpu \
  --max-steps 10
```

For real preprocessed data, set:

```yaml
data:
  kind: pt_manifest
  manifest: /path/to/train_manifest.txt
```

The manifest should contain one `.pt` record path per line. Each record must include `protein_atom_type`, `protein_pos`, `source_atom_type`, `source_pos`, `ligand_atom_type`, and `ligand_pos`; `ligand_bond_edge_index` and `ligand_bond_type` are optional.

Minimal sampling smoke test:

```bash
python3 scripts/sample_diffusion.py \
  --checkpoint outputs/diffusion_tiny/checkpoint_last.pt \
  --out outputs/sample.pt \
  --device cpu \
  --num-steps 32
```
