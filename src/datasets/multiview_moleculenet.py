from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

class MultiviewDataset(Dataset):
    def __init__(self, path):
        self.payload = torch.load(Path(path), map_location='cpu')
        self.records = self.payload['records']
        self.metadata = self.payload['metadata']

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return Data(**self.records[index]).clone()

def scaffold_splits(records, fractions=(0.8, 0.1, 0.1)):
    buckets = {}
    for index, record in enumerate(records):
        mol = Chem.MolFromSmiles(record['smiles'])
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        buckets.setdefault(scaffold, []).append(index)
    groups = sorted(buckets.values(), key=lambda ids: (len(ids), ids[0]), reverse=True)
    n = len(records)
    train, val, test = [], [], []
    for group in groups:
        if len(train) + len(group) <= fractions[0] * n:
            train.extend(group)
        elif len(val) + len(group) <= fractions[1] * n:
            val.extend(group)
        else:
            test.extend(group)

    return {'train': train, 'val': val, 'test': test}

def validate_splits(splits, dataset, require_scaffold=True):
    sets = [set(splits[name]) for name in ['train', 'val', 'test']]
    if any(not s for s in sets) or any(len(s) != len(splits[name]) for s, name in zip(sets, ['train', 'val', 'test'])):
        raise ValueError('Splits must be nonempty without duplicate samples')
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Train, validation and test IDs overlap')
    if set.union(*sets) != set(range(len(dataset))):
        raise ValueError('Split file does not cover exactly the prepared dataset')
    if require_scaffold:
        scaffolds = []
        for ids in sets:
            scaffolds.append({MurckoScaffold.MurckoScaffoldSmiles(
                mol=Chem.MolFromSmiles(dataset.records[i]['smiles']), includeChirality=False) for i in ids})