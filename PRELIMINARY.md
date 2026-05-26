# Protein-Conditioned Full-Molecule Optimization Preliminary

## 任务定位

当前任务不应该再表述成“固定 scaffold 的 ligand generation”。更准确的定位是：

> 在固定 protein pocket 条件下，给定一个低活性或待优化 small molecule，生成/优化得到更高活性的 lead-like molecule；输入分子不再被当作必须固定的 scaffold，而是作为 optimization starting point、source molecule 或 weak constraint。模型需要允许全分子迁移，包括 pose、conformer、substituent、bond/topology，甚至 scaffold migration / scaffold hopping。

因此，主任务是 **protein-conditioned full-molecule optimization / structure-based molecule optimization (SBMO)**，而不是 scaffold-constrained generation。

## 核心变化

旧任务：

```text
context = protein + fixed scaffold
objective = complete or decorate ligand around a preserved scaffold
```

新任务：

```text
context = protein pocket
optional source = low-activity hit / seed molecule
objective = generate improved full molecule under the protein context
```

关键卖点不是“scaffold 保持”，而是：

- 不被固定 scaffold 限制。
- 可以做 scaffold migration / scaffold hopping。
- 仍然利用 protein pocket 保证 target-specific optimization。
- 可以从 low-activity hit 迁移到 high-activity lead-like chemical space。

## 方法路线：diffusion-first

研究问题本身不绑定某一种生成范式，但第一版模型路线先定为 **diffusion-first**。flow matching、BFN、autoregressive / fragment-growth 后续作为 model-family ablation，不进入第一版实现的关键路径。

### Diffusion / score-based models

作为主线实现。DiffSBDD、TargetDiff、DecompDiff、DiffGui、BoKDiff 等工作都说明 diffusion 是 SBDD/SBMO 里的主流路线，尤其适合 3D coordinate denoising、atom/bond co-generation 和 sampling-time guidance。

可行形式：

```text
protein pocket condition
source molecule as initial noisy state / conditional seed / partial prior
reverse diffusion generates optimized full molecule
```

优势：

- 与当前 PocketXMol 的 diffusion/noiser 框架更接近。
- 容易复用 fixed/rigid/flexible scaffold baseline。
- 可以自然加入 Vina/SA/QED/clash/property guidance 或 best-of-k reranking。

风险：

- 离散 atom/bond 与连续坐标需要协同，否则 validity 和 topology 会掉。
- 如果 source molecule conditioning 太强，会退化回 scaffold-constrained generation。

### Flow matching / rectified flow

后续 ablation 路线。

适合 paired low-to-high molecule transition，尤其当数据中能构建明确的 low-activity to high-activity pair。

可行形式：

```text
x0 = low-activity hit state
x1 = high-activity lead state
condition = protein pocket
learn vector field or bridge from x0 to x1
```

优势：

- 任务叙事清晰：hit-to-lead transition。
- 采样路径可解释，适合强调 optimization trajectory。

风险：

- 需要可靠 pairing / atom correspondence / topology transition 设计。
- 如果 pair 不强，容易学成 distribution matching 而不是真实 optimization。

### Bayesian Flow Network / gradient-guided optimization

后续 ablation 或增强路线。

MolCRAFT/MolJO 这条线说明，BFN/continuous parameter space 可以把 continuous coordinates 和 discrete atom types 放进统一可优化空间，并做 gradient guidance。

优势：

- 对 SBMO 的表述很直接：joint optimization of coordinates and discrete types against protein target。
- 可支持 multi-objective、R-group redesign、scaffold hopping 等场景。

风险：

- 需要较多框架迁移成本，不一定适合最快 preliminary。

### Autoregressive / fragment-growth baselines

后续 baseline / ablation 路线。

Pocket2Mol、PocketFlow、ResGen 等可以作为 protein-conditioned de novo generation 或 pocket-specific scaffold exploration baseline。它们不一定是我们的主方法，但很适合作为文献对照和 benchmark baseline。

## 与现有 scaffold preliminary 的关系

原来的 `fixed / rigid / flexible scaffold` 方案保留，但角色改成 baseline / ablation：

- `fixed scaffold`: 传统 scaffold-constrained baseline。
- `rigid scaffold`: 只允许 scaffold pose migration。
- `flexible scaffold`: 允许 scaffold conformer migration。
- `full molecule optimization`: 主方法，允许 atom/type/bond/pose/conformer/scaffold 一起变化。

主实验要证明的是：放开 scaffold 后，模型不是乱生成，而是在 protein context 下更容易到达 active-like chemical space。

## 数据设定

首选测试集可以是 MolGenBench H2L，因为它正好对应 hit-to-lead 场景。但 MolGenBench 不应是唯一 evaluation。

MolGenBench H2L 的推荐组织方式：

```text
input:
  protein pocket
  low-activity hit or seed molecule, depending on method

target/reference:
  high-activity actives or lead molecules from the same target / chemical series
```

如果训练 paired model，需要构建：

```text
(protein, low_activity_mol, high_activity_mol)
```

如果 pairing 质量不够，可以退一步做：

```text
protein-conditioned generation + source-molecule weak conditioning + active-set rediscovery evaluation
```

## Evaluation 不限于 MolGenBench

第一阶段仍然重点看 MolGenBench H2L 的两个指标：

- `Validity`: 确认全分子优化没有牺牲化学有效性。
- `HitRediscover`: 确认模型能迁移到 target-specific active chemical space。

但完整 evaluation 应该分成四层。

### 1. Chemical validity and quality

- Validity
- Uniqueness / diversity
- QED
- SA
- Lipinski / basic drug-likeness filters
- ring / valence / connectivity sanity checks

### 2. Target-conditioned activity proxy

- HitRediscover / ActiveRediscovery
- similarity to known actives
- enrichment among generated candidates
- potency progression if H2L provides low/high activity ordering

### 3. Structure-based binding quality

- Vina Score
- Vina Min
- Vina Dock
- clash / no_clashes
- protein-ligand interaction recovery
- pose reasonableness / docking pose consistency

### 4. Scaffold migration analysis

这层是为了证明我们的优势，而不是常规打分：

- source hit vs generated scaffold similarity
- generated vs active scaffold similarity
- scaffold hopping rate
- matched molecular pair / R-group change statistics
- 是否在 HitRediscover 提升的同时保持足够 scaffold diversity

如果只看 Validity 和 HitRediscover，可能无法证明“骨架迁移”的贡献；如果只看 Vina，又可能忽略 benchmark 的 real-world active rediscovery。因此两类评价都需要。

## 初始实验矩阵

建议 preliminary 先跑 diffusion-first 的四组：

| Group | Condition | Source molecule role | Scaffold constraint | Method family | Purpose |
| --- | --- | --- | --- | --- | --- |
| A0 | protein + scaffold | fixed template | fixed | current maskfill | traditional baseline |
| A1 | protein + scaffold | relaxed template | flexible | current maskfill | relaxed-scaffold baseline |
| B0 | protein + molecule | weak source prior | soft preserve | diffusion | conservative full-mol opt |
| B1 | protein + molecule/protein only | optimization seed or no fixed seed | free migration | diffusion | main full-molecule optimization |

资源有限时先跑：

1. A0: fixed scaffold baseline。
2. B1: full molecule optimization with scaffold migration。

如果 B1 的 validity 崩掉，再加 B0，把 source molecule 作为 soft prior 稳住分布。

## 成功标准

Preliminary 的成功标准不是“某一种方法最好”，而是任务定义成立：

- full-molecule optimization 的 Validity 不明显低于 fixed scaffold baseline。
- HitRediscover / active rediscovery 高于 fixed scaffold baseline。
- 生成分子出现合理 scaffold migration，而不是被原 scaffold 锁死。
- structure-based 指标没有明显恶化，最好 Vina/clash/pose 同步改善。
- 结果不依赖后处理强筛选，而是模型原始生成分布里已有信号。

## 文献定位

当前任务应放在 SBDD/SBMO 的交叉处：

- Pocket2Mol / PocketFlow / ResGen：证明 protein-pocket-conditioned generation 是合理 baseline。
- DiffSBDD / TargetDiff / DecompDiff / DiffGui / BoKDiff：证明 diffusion 是 SBDD 主流路线，尤其适合 3D molecule generation 和 guided optimization。
- MolFORM：说明 flow matching 是后续可做的 ablation 路线。
- MolJO / MolCRAFT：说明 SBMO 的重点是 against protein target 同时优化 continuous coordinates 和 discrete types，并且 scaffold hopping/R-group redesign 是合理应用场景；BFN 可作为后续 model-family comparison。
- MolGenBench / GuacaMol-style rediscovery：说明 Validity 和 Rediscovery/HitRediscover 是必要但不充分的评价；还需要 target-specific binding、drug-likeness 和 scaffold migration analysis。

## 当前一句话版本

我们的 task 是 **protein-conditioned full-molecule hit-to-lead optimization**：在固定 protein pocket 下，从低活性 hit 或 source molecule 出发，第一版使用 diffusion 优化得到高活性 lead-like molecule；与固定 scaffold generation 相比，核心优势是允许 scaffold migration，同时通过 MolGenBench H2L、structure-based docking/proxy metrics 和 scaffold migration analysis 共同验证。flow matching、BFN 等作为后续 ablation。
