import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

class SchNetEncoder(nn.Module):
    def __init__(self, hidden=256, num_layers=4, num_rbf=64, rbf_max=10.0):
        super().__init__()
        self.embedding = nn.Embedding(119, hidden)
        self.register_buffer('centers', torch.linspace(0, rbf_max, num_rbf))
        self.rbf_gamma = (num_rbf / rbf_max) ** 2
        self.filters = nn.ModuleList([nn.Sequential(nn.Linear(num_rbf, hidden), nn.Softplus(),
                                                    nn.Linear(hidden, hidden)) for _ in range(num_layers)])
        self.updates = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.Softplus(),
                                                    nn.Linear(hidden, hidden)) for _ in range(num_layers)])

    def forward(self, data):
        z, pos, batch = data.atomic_numbers, data.pos, data.batch
        pairs = []
        for gid in batch.unique():
            ids = (batch == gid).nonzero(as_tuple=False).flatten()
            src, dst = ids.repeat_interleave(ids.numel()), ids.repeat(ids.numel())
            keep = src != dst
            pairs.append(torch.stack([src[keep], dst[keep]]))
        src, dst = torch.cat(pairs, 1)
        distance = (pos[src] - pos[dst]).norm(dim=-1)
        rbf = torch.exp(-self.rbf_gamma * (distance[:, None] - self.centers[None, :]).square())
        x = self.embedding(z)
        for filt, update in zip(self.filters, self.updates):
            message = x[src] * filt(rbf)
            pooled = torch.zeros_like(x)
            pooled.index_add_(0, dst, message)
            x = x + update(pooled)
        return {'node_emb': x, 'graph_emb': global_mean_pool(x, batch)}