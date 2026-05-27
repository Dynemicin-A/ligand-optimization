# Paper-Grade Pipeline

## 1. Research Task

The task is protein-conditioned full-molecule hit-to-lead optimization:

```text
input:
  protein pocket P
  low-activity source hit M_low

output:
  optimized lead-like molecule M_gen
```

The source hit is a soft condition, not a fixed scaffold. The desired behavior is to improve target-specific active rediscovery and structure-based quality while allowing scaffold migration.

## 2. Related-Work Positioning

The first implementation is diffusion-first because this is the closest family to the current backbone and to several SBDD methods.

| Family | Representative methods | What we borrow | Role |
| --- | --- | --- | --- |
| Diffusion SBDD | DiffSBDD, TargetDiff, DecompDiff, DiffGui, BoKDiff | pocket-conditioned denoising, joint coordinate/type prediction, guidance/rerank | main method |
| Pocket generation | Pocket2Mol, PocketFlow, ResGen | protein-conditioned baselines and sampling protocols | baseline |
| Flow/BFN optimization | MolFORM, FLOWR, MolJO/MolCRAFT | source-to-target transition and gradient-guided SBMO framing | later ablation |
| Benchmarks | MolGenBench H2L, GenBench3D, rediscovery-style benchmarks | Validity, HitRediscover, conformation and rediscovery metrics | evaluation |

## 3. Data Pipeline

Raw H2L data should be normalized into triples:

```text
record_id
protein_path
source_ligand_path
target_ligand_path
target_id / series_id, optional
activity_low / activity_high, optional
```

Executable preprocessing:

```bash
python3 scripts/preprocess_h2l.py \
  --csv data/h2l/train.csv \
  --outdir data/processed_h2l/train \
  --pocket-radius 10
```

Output:

- one `.pt` record per triple
- `manifest.txt` for training

Each record contains:

- `protein_atom_type`, `protein_pos`
- `source_atom_type`, `source_pos`
- `ligand_atom_type`, `ligand_pos`
- optional `ligand_bond_edge_index`, `ligand_bond_type`

## 4. Model Pipeline

Current model:

```text
protein pocket encoder
source-hit encoder
noisy target ligand encoder + time embedding

repeated denoising blocks:
  protein-protein message passing
  source-source message passing
  ligand-ligand message passing
  protein-to-ligand message passing
  source-to-ligand soft message passing
  ligand coordinate denoising update

heads:
  coordinate noise
  atom type logits
  bond type logits
  complex-level score
```

This is intentionally scaffold-migration friendly: no atom mask fixes source atoms in place.

## 5. Training Pipeline

Synthetic smoke training:

```bash
python3 scripts/train_diffusion.py \
  --config configs/train_synthetic_tiny.yaml \
  --outdir outputs/smoke_train \
  --device cpu
```

Real H2L training:

```bash
python3 scripts/train_diffusion.py \
  --config configs/train_h2l_manifest.yaml \
  --outdir outputs/h2l_diffusion \
  --device cuda
```

Losses:

- coordinate noise MSE
- atom type cross entropy
- bond type cross entropy on local ligand edges

Planned additions:

- source-hit preservation/similarity regularizer for conservative B0
- optional property/affinity guidance
- contrastive active-vs-source ranking head

## 6. Sampling Pipeline

Smoke sampling:

```bash
python3 scripts/sample_diffusion.py \
  --checkpoint outputs/h2l_diffusion/checkpoint_last.pt \
  --out outputs/h2l_diffusion/sample.pt \
  --sdf-out outputs/h2l_diffusion/sample.sdf \
  --device cuda \
  --num-steps 128 \
  --num-ligand-atoms 32
```

Current limitation: variable-size proposal and robust molecule reconstruction are not final. The SDF export is a preliminary tensor-to-RDKit conversion for program verification.

## 7. Evaluation Pipeline

Chemical quality:

- Validity
- Uniqueness
- Diversity
- QED
- logP
- molecular weight
- Lipinski pass rate

Active rediscovery / target-conditioned quality:

- exact reference rediscovery
- hit rate by Morgan similarity threshold
- mean max similarity to known actives
- active scaffold recovery

Structure-based quality:

- Vina score/min/dock
- clash/no-clash
- PoseBusters/GenBench3D-style pose sanity
- interaction recovery

Scaffold migration:

- source-vs-generated scaffold similarity
- generated-vs-active scaffold similarity
- scaffold hopping rate
- scaffold diversity among hits

Executable chemical evaluation:

```bash
python3 scripts/evaluate_molecules.py \
  --generated-sdf outputs/h2l_diffusion/samples \
  --reference-smiles data/h2l/reference_actives.smi \
  --source-smiles data/h2l/source_hits.smi \
  --out outputs/h2l_diffusion/eval_metrics.json
```

## 8. Experiment Matrix

| ID | Method | Source use | Scaffold constraint | Purpose |
| --- | --- | --- | --- | --- |
| A0 | PocketXMol maskfill | fixed scaffold | fixed | strongest scaffold-constrained baseline |
| A1 | PocketXMol maskfill | relaxed scaffold | flexible | relaxed-scaffold baseline |
| B0 | diffusion | source as soft prior | soft preserve | conservative full-molecule optimization |
| B1 | diffusion | source as soft condition | free migration | main model |
| C0 | flow matching | low-to-high pair | free migration | later ablation |
| C1 | BFN/MolJO-style | guided optimization | free migration | later ablation |

## 9. Stop Point

Everything up to preprocessing, smoke training, smoke sampling, and chemical evaluation should run locally. The first hard stop is real H2L GPU training and large-scale evaluation, because that requires external data and GPU resources.
