import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pco_backbone.chem import mol_to_record_tensors, parse_pdb_atoms, tensors_to_mol
from pco_backbone.metrics import compute_molecule_metrics, mols_from_smiles


def embed_mol(smiles: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.UFFOptimizeMolecule(mol, maxIters=20)
    return mol


def test_mol_tensor_roundtrip():
    mol = embed_mol("CCO")
    tensors = mol_to_record_tensors(mol)
    restored = tensors_to_mol(
        tensors["atom_type"],
        tensors["pos"],
        tensors["bond_edge_index"],
        tensors["bond_type"],
        sanitize=False,
    )
    assert restored is not None
    assert restored.GetNumAtoms() == mol.GetNumAtoms()


def test_pdb_parser(tmp_path):
    pdb = tmp_path / "pocket.pdb"
    pdb.write_text(
        "ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  O   ALA A   1       1.000   0.000   0.000  1.00 20.00           O\n"
        "END\n"
    )
    protein = parse_pdb_atoms(pdb)
    assert protein["atom_type"].shape == (2,)
    assert protein["pos"].shape == (2, 3)


def test_metrics():
    generated = mols_from_smiles(["CCO", "CCN", "c1ccccc1"])
    refs = mols_from_smiles(["CCO"])
    sources = mols_from_smiles(["CCC"])
    metrics = compute_molecule_metrics(generated, reference_mols=refs, source_mols=sources)
    assert metrics.validity == 1.0
    assert metrics.n_valid == 3
    assert metrics.reference_rediscovery_rate is not None
    assert metrics.reference_hit_rate is not None
    assert metrics.source_similarity_mean is not None
    assert metrics.source_copy_rate is not None
    assert metrics.reference_over_source_similarity_delta is not None

    metrics_with_failure = compute_molecule_metrics(generated + [None], reference_mols=refs, source_mols=sources)
    assert metrics_with_failure.n_total == 4
    assert metrics_with_failure.n_valid == 3
    assert metrics_with_failure.validity == 0.75
