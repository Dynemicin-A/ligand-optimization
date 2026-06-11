# Backbone v3 Reference Plan

更新时间：2026-06-07

本文不是普通文献综述，而是为 ligand optimization backbone v3 选参考架构，并把可落地模块映射到本项目。当前结论：v3 不应该只是在旧原型上继续加深 MLP 或加大 hidden dim，而应该参考当前最强 biomolecular diffusion / flow backbone 的模块组织方式，改成“pair trunk + atom/ligand denoiser + H2L 专用目标”的结构。

当前实验优先级：先跑通 plain v3 baseline，不把 MaskDiT 或 Differential-DiT
机制混进 baseline。MaskDiT 的 structured masked modeling 和 Differential-DiT
的 differential attention 只保留为后续 ablation，等 baseline 完成
PDBbind pretrain + ChEMBL H2L finetune + post-run review 后再逐项启用。

## 1. 参考模型集合

### 1.1 AlphaFold 3

来源：[Nature 2024, Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w)

最值得参考的点：

- 统一处理 protein、nucleic acid、small molecule、ion、modified residue。
- 用 Pairformer 替代 Evoformer，把 pair representation 作为 trunk 的核心信息通道。
- 用 raw atom coordinate diffusion 替代 AF2 的 frame / torsion structure module。
- trunk 和 diffusion module 分离：trunk 学 complex-level pair/single context，diffusion module 在坐标空间迭代 denoise。
- 不强依赖显式 SE(3) equivariant 架构，而是通过数据增强和 diffusion 目标学习几何。
- 训练中使用 confidence / error 预测以及 weighted early stopping，而不是只看单一 loss。
- 对 ligand docking 可加入 pocket-ligand pair token feature。

对本项目的启发：

- 我们当前只有局部 KNN message passing，没有显式 pair trunk。H2L 需要知道 source-target、ligand-pocket、ligand-ligand 的 pair 关系，单纯 node message 不够。
- 应引入 token-level `s_i` 和 pair-level `z_ij` 两套表示，至少做一个 Pairformer-lite。
- diffusion score module 不应该反复重算 protein/source context；context trunk 应该可缓存。
- early stopping 不能只看 `valid_loss`，要看 atom、pose、contact、ranking、hard negative 等组合指标。

### 1.2 Boltz-1 / Boltz-2

来源：

- [Boltz GitHub](https://github.com/jwohlwend/boltz)
- [Boltz-1 technical report PDF](https://gcorso.github.io/assets/boltz1.pdf)
- [Boltz DeepWiki architecture notes](https://deepwiki.com/jwohlwend/boltz/3.1-boltz-1-model)

最值得参考的点：

- 开源 AF3-class biomolecular interaction model，工程实现比 AF3 更容易借鉴。
- 三段式结构清晰：InputEmbedder -> MSA/Pairformer trunk -> AtomDiffusion / Distogram / Confidence heads。
- PairformerModule 同时更新 single (`s`) 和 pair (`z`) 表示。
- AtomDiffusion 负责 3D coordinate generation。
- Distogram head 提供 pairwise distance training signal。
- Confidence module 预测 pLDDT / PAE / PDE 等误差与置信度。
- 计算优化：sequence-local atom attention、attention bias sharing / caching、activation checkpointing、parallel sampling。
- Boltz-2 进一步把 affinity prediction 作为结构预测之外的重要 head。

对本项目的启发：

- v3 需要明确模块边界，而不是把所有 message passing 混在一个 `LigandUpdateBlock` 里。
- 加 `Distogram / contact / clash` pair head，比只靠 coordinate MSE 更稳。
- 加 confidence / ranking head，后续 sampling 时按模型自己估计的 pose/interaction 质量筛样。
- 在 4090 上不能照搬 48-layer Pairformer，但可以做 4-8 层 Pairformer-lite。
- pair bias / local atom attention 的缓存很适合我们长 sampling 时降成本。

### 1.3 Chai-1

来源：[Chai-1 technical report](https://chaiassets.com/chai-1/paper/technical_report_v1.pdf)

最值得参考的点：

- 多模态 biomolecular structure prediction，支持 protein、ligand、nucleic acid 等。
- 架构遵循 AF3-style trunk + diffusion decoder，并大量使用 pair-bias self-attention。
- 支持 MSA、protein language model embedding、ligand SMILES、covalent bond、experimental restraints。
- 可以使用 contact / pocket / restraint 作为提示信号，提高实际结构预测。

对本项目的启发：

- H2L 里 source ligand 应该被当作显式 condition，而不是只丢进一个 source-to-ligand KNN cross message。
- 我们应该为 `source-target alignment`、`source atom is copied / mutated / deleted`、`pocket contact restraint` 设计 feature channel。
- 后续可加入 protein language embedding，但第一版先不要加依赖，先把 pair-bias conditioning 做对。

### 1.4 DiffDock-L

来源：[ICLR 2024, DiffDock-L / confidence bootstrapping](https://proceedings.iclr.cc/paper_files/paper/2024/file/db334db287337b2a365120b524300ef3-Paper-Conference.pdf)

最值得参考的点：

- 证明 docking diffusion 的 generalization 随数据规模和模型容量提升。
- 用 confidence bootstrapping：先 rollout 生成 poses，再用 confidence model 反馈训练 diffusion。
- 训练时保留真实数据，避免 confidence feedback 让模型遗忘最后精修步骤。
- 对 reverse diffusion 的不同噪声阶段施加不同权重。

对本项目的启发：

- 我们 H2L 失败不是简单“跑得不够久”，而是缺少反馈目标。
- 可以做 H2L confidence bootstrapping：生成候选 ligand / pose，按 activity proxy、contact/clash、source similarity、ranking head 过滤，再回灌训练。
- 不要只看 reconstruction loss；要引入 rollout 后的质量反馈。

### 1.5 FlowDock

来源：

- [FlowDock arXiv / Hugging Face paper page](https://huggingface.co/papers/2412.10966)
- [FlowDock paper](https://arxiv.org/abs/2412.10966)

最值得参考的点：

- 用 conditional flow matching 做 flexible protein-ligand docking 和 affinity estimation。
- 学从 unbound / apo structure 到 bound / holo complex 的映射。
- 支持多个 ligand，并把 docking 与 affinity 结合。
- 在 CASP16 ligand affinity prediction 中进入 top-5。

对本项目的启发：

- H2L 本质也可以看成 conditional transport：source low-active complex -> target high-active complex。
- v3 backbone 的接口应该保留 flow matching head，不要把 diffusion 写死。
- pocket 不能永远假设 rigid；即使第一版固定 protein，也要预留 flexible pocket / contact refinement 的模块边界。

### 1.6 MolCRAFT / GeoBFN

来源：

- [MolCRAFT arXiv](https://arxiv.org/abs/2404.12141)
- [MolCRAFT GitHub](https://github.com/GenSI-THUAIR/MolCRAFT)
- [GeoBFN ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/d1b1a091088904cbc7f7faa2b45c8f36-Paper-Conference.pdf)

最值得参考的点：

- 把 3D 坐标和离散 atom type / charge 放到统一连续参数空间建模。
- BFN 通过 Bayesian update 在 parameter space 中逐步生成。
- MolCRAFT 关注 SBDD 中“高 docking 分但 3D pose 不稳定”的 false positive 问题。
- noise-reduced sampling 提高采样质量和速度。

对本项目的启发：

- 我们当前 atom type 是 mask corruption + CE，coordinate 是 Gaussian noise + MSE，两者目标割裂。
- H2L 的 atom identity 一直弱，说明连续坐标和离散化学类型的联合建模不够。
- v3 应把 atom logits、bond logits、coordinate denoising 放进更统一的 head 设计；至少要让 atom/coord/bond 的 conditioning 共享 pair trunk。
- BFN 可以作为 v3 之后的 head family，但现在先抽象接口，避免后面大改。

### 1.7 DecompOpt / DecompDiff / D3FG

来源：

- [DecompOpt ICLR 2024 poster](https://iclr.cc/virtual/2024/poster/18436)
- [DecompDiff ICML 2023](https://proceedings.mlr.press/v202/guan23a.html)
- [D3FG NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/6cdd4ce9330025967dd1ed0bed3010f5-Abstract-Conference.html)

最值得参考的点：

- DecompOpt 直接面向 molecular optimization，而不是纯 de novo generation。
- 用 decomposition 支持 scaffold hopping、R-group optimization、local controllability。
- DecompDiff 将 ligand 分为 arms 和 scaffold，用 decomposed priors。
- D3FG 按 functional group 和 linker 建模，可以做 pocket-specific molecule elaboration。

对本项目的启发：

- H2L 需要 copy / mutate / delete / grow / retype，而不是对整分子统一 denoise。
- v3 应加入 `copy_mutate_gate`：
  - copy source atom / fragment
  - mutate atom type
  - move coordinate
  - add/grow new atom or fragment
  - delete / ignore source region
- 对 changed region 和 unchanged region 分别加 loss；否则模型不知道哪里该保留，哪里该改。

### 1.8 Proteína

来源：[ICLR 2025, Proteína](https://proceedings.iclr.cc/paper_files/paper/2025/file/f4e9121ad30cd4e5528042fbfd835b3f-Paper-Conference.pdf)

最值得参考的点：

- 用 scalable non-equivariant transformer 做 protein structure generation。
- 支持 optional triangle layers，但也展示了 no-triangle model 的可扩展性。
- 通过更大数据和更大模型稳定提升 loss。
- 设计了更适合 protein structure generation 的 time sampling，而不是照搬图像 diffusion。
- 支持 LoRA fine-tuning 和 autoguidance。

对本项目的启发：

- 不必迷信 equivariant GNN；更重要的是 pair/token 表达、训练数据、loss 与 schedule。
- 但是我们当前数据和 GPU 不足以直接上 400M 参数；要做可扩展小模型版本。
- 时间采样要针对局部化学结构和全局 pose 分阶段设计。
- LoRA / adapter 可以作为 H2L finetune 的低风险方式，减少灾难性遗忘。

### 1.9 MolFORM / preference-aligned flow

来源：[MolFORM arXiv](https://arxiv.org/abs/2507.05503)

最值得参考的点：

- 多模态 flow matching，同时处理 discrete atom types 和 continuous coordinates。
- 指出 SBDD generative models 存在 objective misalignment。
- 用 DPO / online reinforcement learning 做 preference alignment，把生成推向高亲和区域。

对本项目的启发：

- 我们已经观察到 H2L 不学，核心就是 objective misalignment。
- H2L 应该有 pairwise preference：target high ligand 比 source low ligand 更好。
- 先实现简单 margin/ranking loss，再考虑 DPO/RL。

### 1.10 SBDD evaluation / practical deployment papers

来源：[ICLR 2025 practical SBDD evaluation](https://proceedings.iclr.cc/paper_files/paper/2025/file/c3ae1bd1ec8a02e1bd3a63a05e14685d-Paper-Conference.pdf)

最值得参考的点：

- Vina score 容易被 atom count、hetero atom、halogen 等因素操纵。
- CrossDocked 数据有 docked structure bias 和 checkpoint selection leakage 风险。
- 评估应加入 known actives similarity、virtual screening、drug-like、SA / selectivity 相关指标。

对本项目的启发：

- 不要把 `valid_loss` 或 Vina score 当最终目标。
- H2L 需要 activity/ranking/similarity/source-improvement 评价。
- 数据 split 要检查 target / series / scaffold leakage。

## 2. v3 Backbone 总体取向

v3 的核心不是“继续加参数”，而是把 backbone 改成下面的结构：

```text
input feature builder
  protein pocket atoms / residues
  source low-active ligand atoms
  noisy target ligand atoms
  optional negative ligand
  source-target alignment / fragment labels
  time / noise level

context trunk, run once per complex where possible
  single representation s_i
  pair representation z_ij
  pair feature embedder: distance, bond, residue/atom type, source-target relation, pocket contact
  Pairformer-lite blocks
  optional triangle-lite / pair mixer

time-conditioned denoiser
  pair-biased ligand/source/protein attention
  atom-local attention / KNN update
  coordinate update
  atom type update
  bond update
  copy/mutate gate

heads
  coordinate denoising / flow vector
  atom logits
  bond logits
  distogram / contact / clash
  source-copy/mutate action logits
  high-vs-low ranking / affinity proxy
  confidence / sample quality
```

## 3. v3 模块设计

### 3.1 Pair Trunk

新增表示：

- `s_i`: token / atom single representation
- `z_ij`: pair representation

`z_ij` 初始特征：

- distance radial basis
- bond type
- same molecule / protein-ligand / source-target relation
- ligand-ligand candidate edge
- source-target nearest atom distance
- optional activity delta / same series flag
- pocket contact prior

模块：

- `PairFeatureEmbedder`
- `PairformerLiteBlock`
- `PairBiasedAttention`
- `PairTransition`
- `TriangleLite` 或 `PairMixer`

第一版不要直接做完整 48 层 Pairformer。建议：

- hidden：`192` 或 `256`
- pair dim：`96` 或 `128`
- layers：`4-8`
- attention：local / block sparse / KNN pair bias
- pair update：先做 cheap pair mixer，triangle attention 作为 ablation

### 3.2 Atom / Ligand Denoiser

当前 `LigandUpdateBlock` 可以保留成 denoiser 的底层局部几何更新，但应该接收 trunk 输出：

- ligand single state
- protein/source context state
- pair bias
- time embedding

denoiser 输出：

- coordinate noise / velocity
- atom logits
- bond logits

保留 diffusion-first，但接口要兼容：

- diffusion noise prediction
- flow matching velocity
- BFN parameter prediction

### 3.3 Source Copy / Mutate Gate

新增 H2L 专用 head：

```text
action per target atom / fragment:
  copy_source
  mutate_atom
  move_atom
  grow_atom
  delete_or_ignore_source
```

训练信号：

- source-target nearest matching
- MCS / substructure alignment
- changed region label
- unchanged region copy consistency
- source similarity regularization

这直接来自 DecompOpt / DecompDiff / D3FG 的思路。我们的 H2L 失败说明普通 denoising 不知道哪里该改。

### 3.4 Hard Negative / Ranking Head

新增 head：

```text
score(complex, ligand)
margin_loss = max(0, margin - score(target_high) + score(source_or_negative_low))
```

负样本：

- 同 series 低活性 ligand
- source ligand
- hard negative ligand
- model-generated bad sample

必须记录：

- `hard_negative_count`
- `hard_negative_loss`
- `positive_score`
- `negative_score`
- `score_gap`

如果这些指标为 0 或不动，直接停，不再长训。

### 3.5 Contact / Clash / Distogram Heads

新增 pair-level auxiliary heads：

- ligand-protein contact logits
- distance bin / distogram
- clash probability
- ligand-ligand bond distance sanity
- optional pocket interaction type

理由：

- PDBbind pretrain 的 coordinate loss 能降，但 H2L pose/atom 信号弱。
- AF3/Boltz 都把 pair distance / confidence 当关键辅助。
- H2L 最终要优化 pocket interaction，不能只做 coordinate MSE。

### 3.6 Confidence / Sample Ranking

新增 confidence head：

- predicted coordinate error
- predicted contact correctness
- predicted clash risk
- predicted high-vs-low improvement
- predicted sample validity

用途：

- sampling 时多样本筛选
- confidence bootstrapping
- early stopping 的 weighted metric

参考 DiffDock-L 和 AF3/Boltz。

## 4. 训练目标重构

v3 loss 不应只由下面三项构成：

```text
pos_loss + atom_loss + bond_loss
```

建议改为：

```text
loss =
  w_pos * coord_loss
  + w_atom * atom_type_loss
  + w_bond * bond_loss
  + w_dist * distogram_loss
  + w_contact * contact_loss
  + w_clash * clash_loss
  + w_copy * copy_mutate_loss
  + w_rank * pairwise_ranking_loss
  + w_conf * confidence_loss
```

训练分阶段：

1. PDBbind / CrossDocked / BindingMOAD pretrain
   - source=target 或 known ligand
   - focus: geometry, atom, bond, contact, distogram

2. ChEMBL H2L supervised finetune
   - source low ligand -> target high ligand
   - focus: copy/mutate, ranking, delta

3. Preference / hard negative finetune
   - generated bad candidates + known actives
   - focus: ranking, contact, clash, sample quality

4. Optional flow / BFN ablation
   - same trunk, different generative head

## 5. 训练日程与早停

参考 AF3 / DiffDock-L / Proteína，v3 训练需要更细的 early-stop。

不能只看 `valid_loss`，应该看 weighted metric：

```text
selection_score =
  valid_loss
  + a * valid_clash
  - b * valid_contact_auc
  - c * score_gap
  + d * atom_type_error
  + e * source_copy_error
```

硬停止条件：

- hard negative loss 始终为 0：停。
- score gap 不扩大：停。
- valid atom loss 持续升高：停。
- H2L 20k step 仍弱于 baseline 且没有趋势：停。
- valid loss 改善但 contact / clash 变差：不要晋级。

checkpoint 策略：

- 每次 validation 如果任一关键 metric best，立即保存对应 checkpoint。
- 不再只按 fixed save interval 保存，避免旧 large run exact best 丢失。

## 6. 推荐实现顺序

### Phase 1：文档和接口

- 定义 `BackboneV3Config`
- 定义 trunk / denoiser / heads 的模块边界
- 训练脚本只接受 v3 config；旧 checkpoint 只能作为离线分析材料，不作为公开接口

### Phase 2：Pair trunk 最小版

新增：

- `PairFeatureEmbedder`
- `PairBiasedAttention`
- `PairformerLiteBlock`
- `DistogramHead`
- `ContactHead`

先不做完整 triangle attention。先做 pair mixer / local pair bias。

### Phase 3：H2L 专用机制

新增：

- `SourceTargetAlignment`
- `CopyMutateGate`
- `RankingHead`
- hard negative sampler 修复
- metrics logging

这一步比继续加大 backbone 更关键。

### Phase 4：短训验证

每个新机制都先跑：

- 2k smoke：loss 是否非零
- 10k signal：train/valid 是否同步改善
- 20k decision：不优于 baseline 就停

不再直接开 12 小时长训。

### Phase 5：sampling / evaluation

补：

- multi-sample ranking
- validity
- source similarity
- scaffold hopping
- contact/clash
- docking proxy
- known active similarity / virtual screening proxy

## 7. 旧原型到 v3 的差距

| 方向 | 旧原型状态 | v3 需要 |
| --- | --- | --- |
| context 表示 | node KNN message passing | single + pair trunk |
| protein-ligand 关系 | cross KNN message | pair bias + contact/distogram |
| source ligand | source-to-ligand message | source-target alignment + copy/mutate gate |
| atom/coord/bond | 三个相对独立 head | 共享 pair trunk 的联合 head |
| hard negative | loss 常为 0 | sampler + ranking head + score gap logging |
| confidence | 无独立 confidence | confidence/sample ranking head |
| early stopping | 主要看 valid_loss | weighted multi-metric selection |
| checkpoint | fixed interval 易丢 exact best | validation best 立即保存 |
| molecule optimization | 类 denoising | pairwise preference / delta optimization |
