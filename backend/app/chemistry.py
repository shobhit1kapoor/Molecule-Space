from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from .config import STRUCTURE_DIM


RDLogger.DisableLog("rdApp.warning")

KNOWN_NAME_TO_SMILES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "naproxen": "COc1ccc2cc([C@H](C)C(=O)O)ccc2c1",
    "diclofenac": "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(C(=O)O)cn1Cc1ccc(F)cc1",
}


def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def canonicalize_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(mol_from_smiles(smiles), canonical=True)


def resolve_name_or_smiles(query: str) -> str:
    text = query.strip()
    if not text:
        raise ValueError("Molecule query is empty")
    known = KNOWN_NAME_TO_SMILES.get(text.lower())
    if known:
        return canonicalize_smiles(known)
    return canonicalize_smiles(text)


def compute_descriptors(smiles: str) -> dict[str, Any]:
    mol = mol_from_smiles(smiles)
    molecular_weight = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rotatable_bonds = int(Lipinski.NumRotatableBonds(mol))
    qed = float(QED.qed(mol))
    lipinski_violations = int(molecular_weight > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    return {
        "molecular_weight": round(molecular_weight, 2),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 2),
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rotatable_bonds,
        "qed": round(qed, 3),
        "lipinski_violations": lipinski_violations,
    }


@lru_cache(maxsize=20000)
def fingerprint_bits(smiles: str) -> tuple[int, ...]:
    mol = mol_from_smiles(smiles)
    bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=STRUCTURE_DIM)
    return tuple(int(bit) for bit in bit_vect.GetOnBits())


def structure_vector(smiles: str) -> list[float]:
    vec = np.zeros((STRUCTURE_DIM,), dtype=np.float32)
    for bit in fingerprint_bits(smiles):
        vec[bit] = 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def sparse_fingerprint(smiles: str) -> tuple[list[int], list[float]]:
    bits = list(fingerprint_bits(smiles))
    return bits, [1.0] * len(bits)


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    mol_a = mol_from_smiles(smiles_a)
    mol_b = mol_from_smiles(smiles_b)
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=STRUCTURE_DIM)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=STRUCTURE_DIM)
    return round(float(DataStructs.TanimotoSimilarity(fp_a, fp_b)), 4)


def molecule_svg(smiles: str, size: tuple[int, int] = (360, 260)) -> str:
    mol = mol_from_smiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace("svg:", "")


def toxicity_flag(descriptors: dict[str, Any]) -> str:
    if descriptors["lipinski_violations"] >= 2 or descriptors["molecular_weight"] > 650 or descriptors["logp"] > 6:
        return "high"
    if descriptors["lipinski_violations"] == 1 or descriptors["logp"] > 4.5 or descriptors["qed"] < 0.35:
        return "medium"
    return "low"
