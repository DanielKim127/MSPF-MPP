import torch
from contextlib import contextmanager
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from src.utils.featurizer import _mol_from_smiles, smiles_to_data


@contextmanager
def _suppress_rdkit_warnings():
    
    RDLogger.DisableLog('rdApp.warning')
    try:
        yield
    finally:
        RDLogger.EnableLog('rdApp.warning')


def molecule_graph(smiles, label=None, gid=None):
    
    mol = _mol_from_smiles(smiles)
    if mol is None:
        raise ValueError('SMILES could not be parsed')
    cleaned = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    graph = smiles_to_data(cleaned, label, gid=gid, inject_pe=False)
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None or mol.GetNumAtoms() != graph.num_nodes:
        raise ValueError('Clean molecule and graph atom counts differ')
    graph.atomic_numbers = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    return graph, mol


def generate_conformer(mol, seed=42, max_iterations=200):
    
    heavy_count = mol.GetNumAtoms()
    with _suppress_rdkit_warnings():
        hydrogenated = Chem.AddHs(mol)
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = int(seed) % 2147483647
        parameters.numThreads = 1
        if AllChem.EmbedMolecule(hydrogenated, parameters) < 0:
            parameters.useRandomCoords = True
            if AllChem.EmbedMolecule(hydrogenated, parameters) < 0:
                raise ValueError('RDKit conformer generation failed')
        if AllChem.MMFFHasAllMoleculeParams(hydrogenated):
            AllChem.MMFFOptimizeMolecule(hydrogenated, maxIters=max_iterations)
        else:
            AllChem.UFFOptimizeMolecule(hydrogenated, maxIters=max_iterations)
        positions = torch.tensor(hydrogenated.GetConformer().GetPositions()[:heavy_count], dtype=torch.float32)
    if not torch.isfinite(positions).all():
        raise ValueError('Invalid conformer coordinates')
    return positions