# Literature Scan for Protein-Conditioned Full-Molecule Optimization

This is a working list, not a final related-work section.

## Diffusion / score-based SBDD

- DiffSBDD: formulates structure-based drug design as 3D conditional generation with an SE(3)-equivariant diffusion model conditioned on protein pockets.
  - https://arxiv.org/abs/2210.13695
- TargetDiff: jointly generates continuous atom coordinates and categorical atom types with an SE(3)-equivariant diffusion model, and also studies affinity prediction/ranking.
  - https://arxiv.org/abs/2303.03543
- DecompDiff: decomposes ligands into scaffold and arms, uses decomposed priors, bond diffusion, and validity guidance.
  - https://arxiv.org/abs/2403.07902
  - https://github.com/bytedance/DecompDiff
- DiffGui: target-aware guided equivariant diffusion with atom/bond diffusion and property guidance.
  - https://www.nature.com/articles/s41467-025-63245-0
- BoKDiff: best-of-k diffusion alignment for target-specific 3D molecule generation.
  - https://arxiv.org/abs/2501.15631

Takeaway: diffusion is not a fallback; it is one of the main families for this task and is probably the closest to the current PocketXMol implementation style.

## Flow matching / BFN / guided optimization

- MolFORM: multi-modal flow matching for SBDD, jointly modeling discrete atom types and continuous 3D coordinates; includes preference alignment with Vina-based reward signals.
  - https://arxiv.org/abs/2507.05503
- FLOWR: continuous and categorical flow matching for structure-aware ligand generation and optimization, including interaction/fragment-based settings.
  - https://arxiv.org/abs/2504.10564
- MolJO / MolCRAFT: gradient-guided Bayesian Flow Networks for SBMO; explicitly frames the task as optimizing continuous coordinates and discrete types against protein targets.
  - https://arxiv.org/abs/2411.13280
  - https://github.com/GenSI-THUAIR/MolCRAFT

Takeaway: flow matching/BFN is useful when we want a clean low-to-high transition story, but it should be compared with diffusion rather than assumed.

## Autoregressive and fragment-growth baselines

- Pocket2Mol: E(3)-equivariant pocket-conditioned molecular sampling, generating atoms/bonds in protein pockets without MCMC.
  - https://arxiv.org/abs/2205.07249
- PocketFlow: structure-based molecular generative model with explicit chemical knowledge inside protein binding pockets.
  - https://www.nature.com/articles/s42256-024-00808-8
- ResGen: commonly used protein-pocket-conditioned 3D molecular generation baseline in later SBDD benchmarks.
  - https://www.nature.com/articles/s42256-023-00712-7

Takeaway: these are useful baselines for target-conditioned generation, even if our final method is non-autoregressive.

## Benchmarks and evaluation

- MolGenBench: large application-oriented benchmark covering de novo design and hit-to-lead optimization; useful for H2L, Validity, and HitRediscover.
  - https://github.com/Intelligent-Drug-Discovery-Lab/MolGenBench
  - https://zenodo.org/records/17890389
- GenBench3D: emphasizes that 3D structure-based molecular generation should evaluate ligand conformation quality inside pockets, not only 2D chemical metrics.
  - https://arxiv.org/abs/2407.04424
- RediscMol: active-molecule rediscovery benchmark, relevant for judging whether generated molecules recover target-relevant active chemical space.
  - https://pubs.acs.org/doi/10.1021/acs.jmedchem.3c02051

Takeaway: MolGenBench H2L is a good first benchmark, but a convincing paper should also include structure-based binding quality and scaffold migration analysis.
