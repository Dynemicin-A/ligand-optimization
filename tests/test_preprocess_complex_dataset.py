import csv
import gzip
import subprocess
import sys
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preprocess_complex_dataset.py"


def write_pdb(path: Path) -> None:
    path.write_text(
        "ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  O   ALA A   1       1.000   0.000   0.000  1.00 20.00           O\n"
        "ATOM      3  N   ALA A   1       0.000   1.000   0.000  1.00 20.00           N\n"
        "END\n"
    )


def write_sdf(path: Path, smiles: str) -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.UFFOptimizeMolecule(mol, maxIters=20)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def run_preprocess(*args: str) -> None:
    subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, check=True)


def test_csv_preprocess_complex_dataset(tmp_path):
    protein = tmp_path / "protein.pdb"
    ligand = tmp_path / "ligand.sdf"
    ligand_gz = tmp_path / "ligand.sdf.gz"
    negative = tmp_path / "negative.sdf"
    write_pdb(protein)
    write_sdf(ligand, "CCO")
    ligand_gz.write_bytes(gzip.compress(ligand.read_bytes()))
    write_sdf(negative, "CCN")
    table = tmp_path / "complexes.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "protein_path", "ligand_path", "negative_ligand_path"])
        writer.writeheader()
        writer.writerow(
            {
                "record_id": "toy",
                "protein_path": protein.name,
                "ligand_path": ligand_gz.name,
                "negative_ligand_path": negative.name,
            }
        )

    outdir = tmp_path / "out_csv"
    run_preprocess("--csv", str(table), "--outdir", str(outdir), "--num-workers", "1")
    manifest = outdir / "manifest.txt"
    paths = [Path(line) for line in manifest.read_text().splitlines() if line]
    assert len(paths) == 1
    record = torch.load(paths[0], map_location="cpu", weights_only=False)
    assert record["record_id"] == "toy"
    assert "negative_ligand_atom_type" in record
    assert "source_edge_index" in record
    assert record["protein_atom_type"].numel() > 0


def test_csv_preprocess_heavy_only_drops_ligand_hydrogens(tmp_path):
    protein = tmp_path / "protein.pdb"
    source = tmp_path / "source.sdf"
    target = tmp_path / "target.sdf"
    negative = tmp_path / "negative.sdf"
    write_pdb(protein)
    write_sdf(source, "CCO")
    write_sdf(target, "CCCO")
    write_sdf(negative, "CCN")
    table = tmp_path / "complexes.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["record_id", "protein_path", "source_ligand_path", "target_ligand_path", "negative_ligand_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "record_id": "toy_heavy",
                "protein_path": protein.name,
                "source_ligand_path": source.name,
                "target_ligand_path": target.name,
                "negative_ligand_path": negative.name,
            }
        )

    outdir = tmp_path / "out_heavy"
    run_preprocess("--csv", str(table), "--outdir", str(outdir), "--num-workers", "1", "--heavy-only")
    path = Path((outdir / "manifest.txt").read_text().strip())
    record = torch.load(path, map_location="cpu", weights_only=False)

    assert record["source_atom_type"].numel() == 3
    assert record["ligand_atom_type"].numel() == 4
    assert record["negative_ligand_atom_type"].numel() == 3
    assert 12 not in record["source_atom_type"].tolist()
    assert 12 not in record["ligand_atom_type"].tolist()
    assert record["hydrogen_policy"] == "heavy_only"


def test_directory_preprocess_multiple_ligands(tmp_path):
    complex_dir = tmp_path / "target_a" / "complex_001"
    complex_dir.mkdir(parents=True)
    write_pdb(complex_dir / "target_pocket.pdb")
    write_sdf(complex_dir / "ligand_a.sdf", "CCO")
    write_sdf(complex_dir / "ligand_b.sdf", "CCN")
    write_sdf(complex_dir / "target_prot.sdf", "CCCC")

    outdir = tmp_path / "out_dir"
    run_preprocess(
        "--root",
        str(tmp_path),
        "--outdir",
        str(outdir),
        "--num-workers",
        "1",
        "--ligands-per-protein",
        "all",
    )
    paths = [Path(line) for line in (outdir / "manifest.txt").read_text().splitlines() if line]
    assert len(paths) == 2
    ids = [torch.load(path, map_location="cpu", weights_only=False)["record_id"] for path in paths]
    assert len(set(ids)) == 2
