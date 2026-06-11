from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass
class MoleculeMetrics:
    n_total: int
    n_valid: int
    validity: float
    uniqueness: float
    qed_mean: float | None
    logp_mean: float | None
    mol_weight_mean: float | None
    lipinski_pass_rate: float | None
    diversity: float | None
    reference_rediscovery_rate: float | None = None
    reference_hit_rate: float | None = None
    reference_similarity_mean: float | None = None
    source_similarity_mean: float | None = None
    source_copy_rate: float | None = None
    reference_over_source_similarity_delta: float | None = None
    scaffold_hopping_rate: float | None = None
    active_scaffold_recovery_rate: float | None = None


def load_smiles(path: str | Path) -> list[str]:
    smiles = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        smiles.append(line.split()[0])
    return smiles


def load_mols_from_sdf(path: str | Path) -> list[Chem.Mol | None]:
    path = Path(path)
    paths = sorted(path.glob("*.sdf")) if path.is_dir() else [path]
    mols: list[Chem.Mol | None] = []
    for sdf in paths:
        supplier = Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=False)
        mols.extend(list(supplier))
    return mols


def mols_from_smiles(smiles: list[str]) -> list[Chem.Mol | None]:
    return [Chem.MolFromSmiles(smi) for smi in smiles]


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)


def morgan_fp(mol: Chem.Mol):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return generator.GetFingerprint(Chem.RemoveHs(mol))


def tanimoto(a: Chem.Mol, b: Chem.Mol) -> float:
    from rdkit.DataStructs import TanimotoSimilarity

    return float(TanimotoSimilarity(morgan_fp(a), morgan_fp(b)))


def lipinski_pass(mol: Chem.Mol) -> bool:
    return (
        Descriptors.MolWt(mol) <= 500
        and Crippen.MolLogP(mol) <= 5
        and Lipinski.NumHDonors(mol) <= 5
        and Lipinski.NumHAcceptors(mol) <= 10
    )


def scaffold_smiles(mol: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(Chem.RemoveHs(mol))
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold, canonical=True)


def pairwise_diversity(valid_mols: list[Chem.Mol], max_pairs: int = 5_000) -> float | None:
    if len(valid_mols) < 2:
        return None
    sims = []
    n_pairs = 0
    for i in range(len(valid_mols)):
        for j in range(i + 1, len(valid_mols)):
            sims.append(tanimoto(valid_mols[i], valid_mols[j]))
            n_pairs += 1
            if n_pairs >= max_pairs:
                return 1.0 - mean(sims)
    return 1.0 - mean(sims)


def compute_molecule_metrics(
    mols: list[Chem.Mol | None],
    *,
    reference_mols: list[Chem.Mol | None] | None = None,
    source_mols: list[Chem.Mol | None] | None = None,
    rediscovery_similarity_threshold: float = 0.7,
    scaffold_hopping_similarity_threshold: float = 0.4,
) -> MoleculeMetrics:
    valid_mols = [mol for mol in mols if mol is not None]
    n_total = len(mols)
    n_valid = len(valid_mols)
    valid_smiles = [canonical_smiles(mol) for mol in valid_mols]
    unique_smiles = set(valid_smiles)

    ref_valid = [mol for mol in reference_mols or [] if mol is not None]
    src_valid = [mol for mol in source_mols or [] if mol is not None]
    ref_smiles = {canonical_smiles(mol) for mol in ref_valid}
    ref_scaffolds = {scaffold_smiles(mol) for mol in ref_valid}
    ref_scaffolds.discard("")

    reference_rediscovery_rate = None
    reference_hit_rate = None
    reference_similarity_mean = None
    active_scaffold_recovery_rate = None
    if ref_valid and valid_mols:
        exact_hits = [smi in ref_smiles for smi in valid_smiles]
        reference_rediscovery_rate = sum(exact_hits) / len(valid_smiles)
        max_sims = [max(tanimoto(mol, ref) for ref in ref_valid) for mol in valid_mols]
        reference_hit_rate = sum(sim >= rediscovery_similarity_threshold for sim in max_sims) / len(max_sims)
        reference_similarity_mean = mean(max_sims)
        gen_scaffolds = [scaffold_smiles(mol) for mol in valid_mols]
        active_scaffold_recovery_rate = sum(scaf in ref_scaffolds for scaf in gen_scaffolds) / len(gen_scaffolds)

    scaffold_hopping_rate = None
    source_similarity_mean = None
    source_copy_rate = None
    reference_over_source_similarity_delta = None
    if src_valid and valid_mols:
        source_scaffolds = [scaffold_smiles(mol) for mol in src_valid]
        source_sims = []
        hop_flags = []
        copy_flags = []
        for i, mol in enumerate(valid_mols):
            src = src_valid[min(i, len(src_valid) - 1)]
            source_sim = tanimoto(mol, src)
            source_sims.append(source_sim)
            same_scaffold = scaffold_smiles(mol) == source_scaffolds[min(i, len(source_scaffolds) - 1)]
            similar_to_source = source_sim >= scaffold_hopping_similarity_threshold
            copy_flags.append(canonical_smiles(mol) == canonical_smiles(src))
            hop_flags.append((not same_scaffold) and (not similar_to_source))
        source_similarity_mean = mean(source_sims)
        source_copy_rate = sum(copy_flags) / len(copy_flags)
        scaffold_hopping_rate = sum(hop_flags) / len(hop_flags)
        if reference_similarity_mean is not None:
            reference_over_source_similarity_delta = reference_similarity_mean - source_similarity_mean

    return MoleculeMetrics(
        n_total=n_total,
        n_valid=n_valid,
        validity=n_valid / max(n_total, 1),
        uniqueness=len(unique_smiles) / max(n_valid, 1) if n_valid else 0.0,
        qed_mean=mean(QED.qed(mol) for mol in valid_mols) if valid_mols else None,
        logp_mean=mean(Crippen.MolLogP(mol) for mol in valid_mols) if valid_mols else None,
        mol_weight_mean=mean(Descriptors.MolWt(mol) for mol in valid_mols) if valid_mols else None,
        lipinski_pass_rate=mean(1.0 if lipinski_pass(mol) else 0.0 for mol in valid_mols) if valid_mols else None,
        diversity=pairwise_diversity(valid_mols),
        reference_rediscovery_rate=reference_rediscovery_rate,
        reference_hit_rate=reference_hit_rate,
        reference_similarity_mean=reference_similarity_mean,
        source_similarity_mean=source_similarity_mean,
        source_copy_rate=source_copy_rate,
        reference_over_source_similarity_delta=reference_over_source_similarity_delta,
        scaffold_hopping_rate=scaffold_hopping_rate,
        active_scaffold_recovery_rate=active_scaffold_recovery_rate,
    )
