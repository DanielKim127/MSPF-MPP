from typing import List
import os
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data


PE_MODE = os.getenv("PE_MODE", "none").lower()   
GMPE_CACHE_DIR = os.getenv("GMPE_CACHE_DIR", r"")  
PE_DIM = int(os.getenv("PE_DIM", "0"))           

ATOM_LIST = list(range(1, 119))

def one_hot(x, allowable):
    return [int(x == value) for value in allowable]

def atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    return (
        one_hot(atom.GetAtomicNum(), ATOM_LIST) +
        one_hot(int(atom.GetHybridization()), [
            int(Chem.rdchem.HybridizationType.SP),
            int(Chem.rdchem.HybridizationType.SP2),
            int(Chem.rdchem.HybridizationType.SP3),
            int(Chem.rdchem.HybridizationType.SP3D),
            int(Chem.rdchem.HybridizationType.SP3D2)
        ]) +
        [atom.GetTotalNumHs(includeNeighbors=True),
         atom.GetFormalCharge(),
         int(atom.GetIsAromatic())]
    )

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


STEREO_CATS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]
STEREO2IDX = {s: i for i, s in enumerate(STEREO_CATS)}
EDGE_DIM = 4 + 1 + 1 + 6  

def bond_features(bond: Chem.rdchem.Bond | None) -> List[float]:
    if bond is None:
        return [0.0] * EDGE_DIM
    
    type_ohe = [float(bond.GetBondType() == bond_type) for bond_type in BOND_TYPES]
    
    is_conj = float(bond.GetIsConjugated())
    in_ring = float(bond.IsInRing())
    
    st = bond.GetStereo()
    stereo_ohe = [0.0] * len(STEREO_CATS)
    stereo_ohe[STEREO2IDX.get(st, 0)] = 1.0
    return type_ohe + [is_conj] + [in_ring] + stereo_ohe


def _mol_from_smiles(smi: str):
    return Chem.MolFromSmiles(smi)

def smiles_to_data(smiles: str, y_vec, gid=None, inject_pe: bool = True) -> Data:
    
    mol = _mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_index.append([i, j]); edge_attr.append(bf)
        edge_index.append([j, i]); edge_attr.append(bf)
    if len(edge_index) == 0:
        edge_index = [[0, 0]]
        edge_attr = [bond_features(None)]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    
    if y_vec is None:
        y = None
    else:
        arr = np.asarray(y_vec)
        if arr.ndim == 0:
            y = torch.tensor([[float(arr)]], dtype=torch.float)
        else:
            y = torch.tensor(arr, dtype=torch.float).reshape(1, -1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = y
    data.smiles = smiles
    if gid is not None:
        data.idx = int(gid)

    
    if inject_pe:
        if PE_MODE == "rwpe":
            from src.transforms.RWPE import RandomWalkPE
            K = PE_DIM if PE_DIM > 0 else 16
            data = RandomWalkPE(K=K, normalise=True)(data)

        elif PE_MODE == "gmpe":
            gmpepath = os.path.join(GMPE_CACHE_DIR, f"{int(gid)}.pt")
            pe_gmpe = torch.load(gmpepath, map_location="cpu")["pe"].float()
            if pe_gmpe.dim() == 1:
                pe_gmpe = pe_gmpe.unsqueeze(1)
            data.pe = pe_gmpe
            data.idx = int(gid)

    return data