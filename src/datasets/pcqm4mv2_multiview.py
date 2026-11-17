import json
from collections import OrderedDict
from pathlib import Path
import bisect
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class PairedMoleculeDataset(Dataset):
    def __init__(self, root, cache_shards=2, mmap_shards=False):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        self.shards = self.manifest['shards']
        self.ends, total = [], 0
        for shard in self.shards:
            total += shard['count']
            self.ends.append(total)
        self.cache, self.cache_shards = OrderedDict(), max(1, int(cache_shards))
        self.mmap_shards = bool(mmap_shards)

    def shard_for_index(self, index):
        return bisect.bisect_right(self.ends, index)

    def _load_shard(self, shard_index):
        shard = self.cache.get(shard_index)
        if shard is None:
            path = self.root / self.shards[shard_index]['file']
            kwargs = {'map_location': 'cpu', 'weights_only': False}
            if self.mmap_shards:
                kwargs['mmap'] = True
            shard = torch.load(path, **kwargs)
            self.cache[shard_index] = shard
            while len(self.cache) > self.cache_shards:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(shard_index)
        return shard

    def _item_from_shard(self, index, shard_index, shard):
        start = self.ends[shard_index - 1] if shard_index else 0
        
        return Data(**shard[index - start])

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = self.shard_for_index(index)
        return self._item_from_shard(index, shard_index, self._load_shard(shard_index))

    def __getitems__(self, indices):     
        grouped = {}
        for position, index in enumerate(indices):
            if index < 0 or index >= len(self):
                raise IndexError(index)
            grouped.setdefault(self.shard_for_index(index), []).append((position, index))
        items = [None] * len(indices)
        for shard_index, positions in grouped.items():
            shard = self._load_shard(shard_index)
            for position, index in positions:
                items[position] = self._item_from_shard(index, shard_index, shard)
        return items