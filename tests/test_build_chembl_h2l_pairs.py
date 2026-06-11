import csv
import subprocess
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_chembl_h2l_pairs.py"


def write_sdf(path: Path, smiles: str) -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=11)
    AllChem.UFFOptimizeMolecule(mol, maxIters=20)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def write_pdb(path: Path) -> None:
    path.write_text(
        "ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  O   ALA A   1       1.000   0.000   0.000  1.00 20.00           O\n"
        "END\n"
    )


def test_pair_builder_filters_large_structure_jumps(tmp_path):
    round_dir = tmp_path / "round1"
    series_dir = round_dir / "train" / "S1"
    series_dir.mkdir(parents=True)
    write_pdb(series_dir / "S1_prep.pdb")
    write_sdf(series_dir / "low.sdf", "CCO")
    write_sdf(series_dir / "near_high.sdf", "CCCO")
    write_sdf(series_dir / "far_high.sdf", "c1ccccc1")
    csv_path = round_dir / "train.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sries_id", "identifier", "affinity", "affinity_units", "affinity_type", "pdbid", "uniprot"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sries_id": "S1",
                "identifier": "low",
                "affinity": "1000",
                "affinity_units": "nM",
                "affinity_type": "IC50",
                "pdbid": "P1",
                "uniprot": "U1",
            }
        )
        writer.writerow(
            {
                "sries_id": "S1",
                "identifier": "near_high",
                "affinity": "100",
                "affinity_units": "nM",
                "affinity_type": "IC50",
                "pdbid": "P1",
                "uniprot": "U1",
            }
        )
        writer.writerow(
            {
                "sries_id": "S1",
                "identifier": "far_high",
                "affinity": "10",
                "affinity_units": "nM",
                "affinity_type": "IC50",
                "pdbid": "P1",
                "uniprot": "U1",
            }
        )

    outdir = tmp_path / "pairs"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--outdir",
            str(outdir),
            "--include-splits",
            "train",
            "--min-delta",
            "0.5",
            "--min-tanimoto",
            "0.35",
            "--max-heavy-delta",
            "3",
        ],
        cwd=ROOT,
        check=True,
    )
    with (outdir / "pairs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert any(row["target_identifier"] == "near_high" for row in rows)
    assert all(row["target_identifier"] != "far_high" for row in rows)
    assert all(float(row["tanimoto"]) >= 0.35 for row in rows)
