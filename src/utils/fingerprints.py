import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

FP_DIM = 1048
FP_SCHEMA = 'rdkit-maccs167+pybiomed-pubchem881-v1'


def fingerprint(smiles):
    try:
        from PyBioMed.PyMolecule.PubChemFingerprints import calcPubChemFingerAll
    except ImportError as exc:
        raise ImportError('import error') from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    maccs = np.zeros(167, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs)
    pubchem = np.asarray(calcPubChemFingerAll(mol), dtype=np.int8)
    return torch.from_numpy(np.concatenate([maccs, pubchem]).astype(np.float32)).reshape(1, FP_DIM)