import copy
import json
import random
import hashlib
import bisect
from pathlib import Path
import numpy as np
import torch
import yaml
from src.models.DSPGNN import DSPGNN
from src.models.schnet import SchNetEncoder
from src.transforms.GMPE.encoder import SiameseGNN

ROOT = Path(__file__).resolve().parents[2]

def path_from_root(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    with open(path_from_root(path), encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(value='auto'):
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    if value == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    return torch.device(value)


def build_dspgnn(config):
    cfg = copy.deepcopy(config)
    cfg['out_dim'] = None
    return DSPGNN(**cfg)


def build_schnet(config):
    return SchNetEncoder(**config)


def gmpe_payload(checkpoint):
    blob = torch.load(path_from_root(checkpoint), map_location='cpu', weights_only=False)
    state, metadata = blob['model'], blob['args']
    weight = state['backbone.node_in.weight']
    layers = len({key.split('.')[2] for key in state if key.startswith('backbone.convs.')})
    edge = state['backbone.edge_in.weight']
    config = dict(in_dim=weight.size(1), edge_dim=edge.size(1),
                  hidden=weight.size(0), layers=layers, dropout=float(metadata['dropout']),
                  use_edge=True, normalize_emb=bool(metadata['normalize_emb']))
    return {'config': config, 'state': state}


class GMPEGenerator:
    def __init__(self, payload):
        self.payload = payload
        self.model = SiameseGNN(**payload['config']).cpu().eval()
        self.model.load_state_dict(payload['state'], strict=True)

    @torch.no_grad()
    def __call__(self, data):
        return self.model.encode_single_graph(data.cpu()).float()


def tensor_record(data):
    return {key: value.cpu() if isinstance(value, torch.Tensor) else value
            for key, value in data.to_dict().items()}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite_gradients(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f'Nonfinite gradient: {name}')


class PairedBatchSampler:
    def __init__(self, size, batch_size, shuffle=False, seed=42):
        self.size, self.batch_size, self.shuffle, self.seed = size, batch_size, shuffle, seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist() if self.shuffle else list(range(self.size))
        chunks = [indices[i:i + self.batch_size] for i in range(0, self.size, self.batch_size)]
        if len(chunks) > 1 and len(chunks[-1]) == 1:
            chunks[-2].extend(chunks.pop())
        yield from chunks

    def __len__(self):
        count = (self.size + self.batch_size - 1) // self.batch_size
        return count - (count > 1 and self.size % self.batch_size == 1)


class ShardAwarePairedBatchSampler:
    def __init__(self, indices, shard_ends, batch_size, shuffle=False, seed=42):
        self.indices = list(indices)
        self.shard_ends = list(shard_ends)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.groups = [[] for _ in self.shard_ends]
        for index in self.indices:
            shard = bisect.bisect_right(self.shard_ends, index)
            self.groups[shard].append(index)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shard_order = torch.randperm(len(self.groups), generator=generator).tolist() if self.shuffle else range(len(self.groups))
        carry, batches = [], []
        for shard_index in shard_order:
            group = self.groups[shard_index]
            if not group:
                continue
            if self.shuffle and len(group) > 1:
                order = torch.randperm(len(group), generator=generator).tolist()
                group = [group[i] for i in order]
            if carry:
                take = min(self.batch_size - len(carry), len(group))
                carry.extend(group[:take])
                group = group[take:]
                if len(carry) == self.batch_size:
                    batches.append(carry)
                    carry = []
            full = len(group) // self.batch_size
            batches.extend(group[i * self.batch_size:(i + 1) * self.batch_size] for i in range(full))
            carry.extend(group[full * self.batch_size:])
        if carry:
            if len(carry) == 1 and batches:
                batches[-1].extend(carry)
            else:
                batches.append(carry)
        yield from batches

    def __len__(self):
        count = (len(self.indices) + self.batch_size - 1) // self.batch_size
        return count - (count > 1 and len(self.indices) % self.batch_size == 1)