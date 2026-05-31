from __future__ import annotations

import gzip
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from rdkit import Chem


DEFAULT_LIGAND_ATOMS = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "P",
    "Cl",
    "Br",
    "I",
    "B",
    "Si",
    "Se",
    "H",
    "Na",
    "K",
    "*",
]
DEFAULT_PROTEIN_ATOMS = [
    "C",
    "N",
    "O",
    "S",
    "P",
    "H",
    "F",
    "Cl",
    "Br",
    "I",
    "B",
    "Se",
    "Zn",
    "Mg",
    "Ca",
    "Fe",
    "Mn",
    "Cu",
    "Co",
    "Ni",
    "Na",
    "K",
    "Si",
    "*",
]

BOND_TYPE_TO_RDKIT = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}
RDKIT_BOND_TO_TYPE = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}


@dataclass(frozen=True)
class AtomVocab:
    atoms: tuple[str, ...]

    @classmethod
    def ligand_default(cls) -> "AtomVocab":
        return cls(tuple(DEFAULT_LIGAND_ATOMS))

    @classmethod
    def protein_default(cls) -> "AtomVocab":
        return cls(tuple(DEFAULT_PROTEIN_ATOMS))

    @property
    def unknown_index(self) -> int:
        return len(self.atoms) - 1

    def encode(self, symbol: str) -> int:
        symbol = normalize_element(symbol)
        try:
            return self.atoms.index(symbol)
        except ValueError:
            return self.unknown_index

    def decode(self, index: int) -> str:
        if index < 0 or index >= len(self.atoms):
            return "*"
        symbol = self.atoms[index]
        return "C" if symbol == "*" else symbol


def normalize_element(raw: str) -> str:
    raw = "".join(ch for ch in raw.strip() if ch.isalpha())
    if not raw:
        return "*"
    if len(raw) == 1:
        return raw.upper()
    return raw[0].upper() + raw[1:].lower()


def load_first_mol(path: str | Path, sanitize: bool = True) -> Chem.Mol:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".gz":
        inner_suffix = Path(path.stem).suffix or ".sdf"
        handle = tempfile.NamedTemporaryFile(suffix=inner_suffix, delete=False)
        tmp_path = Path(handle.name)
        try:
            with gzip.open(path, "rb") as source:
                handle.write(source.read())
            handle.close()
            return load_first_mol(tmp_path, sanitize=sanitize)
        finally:
            handle.close()
            tmp_path.unlink(missing_ok=True)
    if suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), sanitize=sanitize, removeHs=False)
        for mol in supplier:
            if mol is not None:
                return mol
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), sanitize=sanitize, removeHs=False)
        if mol is not None:
            return mol
    elif suffix in {".smi", ".smiles", ".txt"}:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                mol = Chem.MolFromSmiles(line.split()[0], sanitize=sanitize)
                if mol is not None:
                    return mol
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), sanitize=sanitize, removeHs=False)
        if mol is not None:
            return mol
    raise ValueError(f"could not load molecule from {path}")


def mol_to_record_tensors(mol: Chem.Mol, vocab: AtomVocab | None = None) -> dict[str, torch.Tensor]:
    vocab = vocab or AtomVocab.ligand_default()
    conf = mol.GetConformer() if mol.GetNumConformers() else None
    atom_type = []
    pos = []
    for atom in mol.GetAtoms():
        atom_type.append(vocab.encode(atom.GetSymbol()))
        if conf is None:
            pos.append([0.0, 0.0, 0.0])
        else:
            p = conf.GetAtomPosition(atom.GetIdx())
            pos.append([p.x, p.y, p.z])

    bond_edges = []
    bond_types = []
    for bond in mol.GetBonds():
        bond_edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        bond_types.append(RDKIT_BOND_TO_TYPE.get(bond.GetBondType(), 1))

    return {
        "atom_type": torch.tensor(atom_type, dtype=torch.long),
        "pos": torch.tensor(pos, dtype=torch.float32),
        "bond_edge_index": (
            torch.tensor(bond_edges, dtype=torch.long).t().contiguous()
            if bond_edges
            else torch.empty(2, 0, dtype=torch.long)
        ),
        "bond_type": torch.tensor(bond_types, dtype=torch.long),
    }


def parse_pdb_atoms(path: str | Path, vocab: AtomVocab | None = None) -> dict[str, torch.Tensor]:
    vocab = vocab or AtomVocab.protein_default()
    atom_type = []
    pos = []
    with Path(path).open("r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            element = line[76:78].strip() if len(line) >= 78 else ""
            if not element:
                element = line[12:16].strip()
            atom_type.append(vocab.encode(element))
            pos.append([x, y, z])
    if not atom_type:
        raise ValueError(f"no ATOM/HETATM coordinates found in {path}")
    return {
        "atom_type": torch.tensor(atom_type, dtype=torch.long),
        "pos": torch.tensor(pos, dtype=torch.float32),
    }


def crop_protein_to_ligand(
    protein: dict[str, torch.Tensor],
    ligand_pos: torch.Tensor,
    radius: float,
) -> dict[str, torch.Tensor]:
    if radius <= 0:
        return protein
    dist = torch.cdist(protein["pos"], ligand_pos)
    keep = dist.min(dim=1).values <= radius
    if not keep.any():
        return protein
    return {
        "atom_type": protein["atom_type"][keep],
        "pos": protein["pos"][keep],
    }


def tensors_to_mol(
    atom_type: torch.Tensor,
    pos: torch.Tensor,
    bond_edge_index: torch.Tensor,
    bond_type: torch.Tensor,
    vocab: AtomVocab | None = None,
    sanitize: bool = True,
) -> Chem.Mol | None:
    vocab = vocab or AtomVocab.ligand_default()
    mol = Chem.RWMol()
    for idx in atom_type.detach().cpu().long().tolist():
        symbol = vocab.decode(idx)
        if symbol == "H":
            symbol = "C"
        mol.AddAtom(Chem.Atom(symbol))

    seen = set()
    edge_cpu = bond_edge_index.detach().cpu().long()
    bond_cpu = bond_type.detach().cpu().long()
    for i in range(edge_cpu.shape[1]):
        a = int(edge_cpu[0, i])
        b = int(edge_cpu[1, i])
        if a == b or a < 0 or b < 0 or a >= mol.GetNumAtoms() or b >= mol.GetNumAtoms():
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        bt = int(bond_cpu[i]) if i < bond_cpu.shape[0] else 1
        if bt <= 0:
            continue
        mol.AddBond(a, b, BOND_TYPE_TO_RDKIT.get(bt, Chem.BondType.SINGLE))

    out = mol.GetMol()
    conf = Chem.Conformer(out.GetNumAtoms())
    pos_cpu = pos.detach().cpu().float()
    for i in range(out.GetNumAtoms()):
        xyz = pos_cpu[i].tolist()
        conf.SetAtomPosition(i, xyz)
    out.AddConformer(conf)

    if sanitize:
        try:
            Chem.SanitizeMol(out)
        except Exception:
            return None
    return out


def write_mol_sdf(mol: Chem.Mol, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
