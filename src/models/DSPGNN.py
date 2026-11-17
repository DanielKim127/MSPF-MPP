import math
from typing import Optional
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import Set2Set, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax
from torch.utils.checkpoint import checkpoint

class DSPGNNLayer(nn.Module):
    def __init__(self, hidden, heads, edge_dim, dropout=0.5):
        super().__init__()
        if hidden % heads:
            raise ValueError('hidden must be divisible by heads')
        self.heads, self.dim = heads, hidden // heads
        self.qkv = nn.ModuleDict({f'{channel}_{kind}': nn.ModuleList([
            nn.Linear(2 * hidden if channel == 'h' else hidden, hidden, bias=False)
            for _ in range(3)]) for channel in ['h', 'p'] for kind in ['bond', 'virtual']})
        self.edge_bias = nn.ModuleDict({f'{channel}_{kind}': nn.Linear(edge_dim, heads)
                                       for channel in ['h', 'p'] for kind in ['bond', 'virtual']})
        self.edge_value = nn.Linear(edge_dim, hidden, bias=False)
        self.edge_gate = nn.Parameter(torch.zeros(heads))
        self.out_h, self.out_p = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * hidden, hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, data):
        h, p = data.x, data.pe
        ei, e, real = data.attention_edge_index, data.attention_edge_attr, data.real_mask
        src, dst = ei
        outputs, caches = {}, {}
        for channel, inputs in [('h', torch.cat([h, p], -1)), ('p', p)]:
            scores = h.new_empty((ei.size(1), self.heads))
            values = h.new_empty((ei.size(1), self.heads, self.dim))
            for kind, mask in [('bond', real), ('virtual', ~real)]:
                q, k, v = [linear(inputs).view(-1, self.heads, self.dim)
                           for linear in self.qkv[f'{channel}_{kind}']]
                scores[mask] = (q[dst[mask]] * k[src[mask]]).sum(-1) / math.sqrt(self.dim)
                scores[mask] = scores[mask] + self.edge_bias[f'{channel}_{kind}'](e[mask])
                values[mask] = v[src[mask]]
            if channel == 'h':
                bond_values = self.edge_value(e).view(-1, self.heads, self.dim)
                values = values + bond_values * self.edge_gate.tanh()[None, :, None] * real[:, None, None]
            attention = softmax(scores, dst, num_nodes=h.size(0))
            caches[channel] = attention.detach()
            message = self.drop(attention).unsqueeze(-1) * values
            output = h.new_zeros((h.size(0), self.heads, self.dim))
            output.index_add_(0, dst, message)
            outputs[channel] = output.flatten(1)
        h = self.norm1(h + self.drop(self.out_h(outputs['h'])))
        data.x = self.norm2(h + self.drop(self.ffn(h)))
        data.pe = p + torch.tanh(self.drop(self.out_p(outputs['p'])))
        self.attn_cache = caches['h']
        self.edge_index_cache = ei.detach()
        return data

def add_attention_edges(data):
    batch = getattr(data, 'batch', None)
    if batch is None:
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        data.batch = batch
    edges = []
    for gid in batch.unique():
        nodes = (batch == gid).nonzero(as_tuple=False).flatten()
        src = nodes.repeat_interleave(nodes.numel())
        dst = nodes.repeat(nodes.numel())
        keep = (src != dst) if nodes.numel() > 1 else torch.ones_like(src, dtype=torch.bool)
        edges.append(torch.stack([src[keep], dst[keep]]))
    ei = torch.cat(edges, dim=1)
    n = data.num_nodes
    bond_keys = data.edge_index[0] * n + data.edge_index[1]
    order = bond_keys.argsort()
    sorted_keys = bond_keys[order]
    keys = ei[0] * n + ei[1]
    positions = torch.searchsorted(sorted_keys, keys)
    safe = positions.clamp_max(max(0, sorted_keys.numel() - 1))
    real = (positions < sorted_keys.numel()) & (sorted_keys[safe] == keys)
    attr = data.edge_attr.new_zeros((ei.size(1), data.edge_attr.size(1)))
    attr[real] = data.edge_attr[order[safe[real]]]
    data.attention_edge_index, data.attention_edge_attr, data.real_mask = ei, attr, real
    return data

class DSPGNN(nn.Module):
    def __init__(
        self,
        in_dim: int = 126,
        hidden: int = 256,
        pe_dim: int = 128,
        num_layers: int = 5,
        num_heads: int = 4,
        edge_dim: int = 12,
        dropout: float = 0.5,
        readout: str = 'mean',
        out_dim: Optional[int] = None,
        s2s_steps: int = 3,
        s2s_layers: int = 3,
    ):
        super().__init__()
        if hidden % num_heads:
            raise ValueError('hidden must be divisible by num_heads')
        if readout not in {'mean', 'sum', 'max', 'set2set'}:
            raise ValueError('readout must be mean, sum, max or set2set')
        self.hidden = hidden
        self.readout_type = readout
        self.gradient_checkpointing = False
        self.node_in = nn.Linear(in_dim, hidden)
        self.pe_proj = nn.Linear(pe_dim, hidden) if pe_dim != hidden else nn.Identity()
        self.pe_norm = nn.LayerNorm(hidden)
        self.pe_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DSPGNNLayer(hidden, num_heads, edge_dim, dropout) for _ in range(num_layers)
        ])
        if readout == 'set2set':
            self.s2s = Set2Set(hidden, processing_steps=s2s_steps, num_layers=s2s_layers)
            readout_dim = hidden * 2
        else:
            self.s2s = None
            readout_dim = hidden
        self.head = None if out_dim is None else nn.Sequential(
            nn.Linear(readout_dim, 300, bias=True),
            nn.LayerNorm(300),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(300, out_dim, bias=True),
        )

    def graph_readout(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.readout_type == 'mean':
            return global_mean_pool(x, batch)
        if self.readout_type == 'sum':
            return global_add_pool(x, batch)
        if self.readout_type == 'max':
            return global_max_pool(x, batch)
        return self.s2s(x, batch)

    def forward(self, data):
        graph = self.encode(data)['graph_emb']
        return self.head(graph) if self.head is not None else graph

    def forward_features(self, data, return_node_emb: bool = False):
        features = self.encode(data)
        return features['node_emb' if return_node_emb else 'graph_emb']

    def encode(self, data):
        data = data.clone()
        data.x = self.node_in(data.x)
        data.pe = self.pe_drop(self.pe_norm(self.pe_proj(data.pe)))
        data = add_attention_edges(data)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                def layer_forward(x, pe, ei, edge_attr, real_mask, current_layer=layer):
                    layer_data = Data(x=x, pe=pe, attention_edge_index=ei,
                                      attention_edge_attr=edge_attr, real_mask=real_mask)
                    transformed = current_layer(layer_data)
                    return transformed.x, transformed.pe

                data.x, data.pe = checkpoint(
                    layer_forward, data.x, data.pe, data.attention_edge_index,
                    data.attention_edge_attr, data.real_mask,
                    use_reentrant=False, preserve_rng_state=True,
                )
            else:
                data = layer(data)
        return {
            'node_emb': data.x,
            'graph_emb': self.graph_readout(data.x, data.batch),
            'position_emb': data.pe,
        }

    def set_finetune(self, last_k_layers=2):
        if last_k_layers < 0:
            self.requires_grad_(True)
            return list(self.parameters())
        for parameter in self.parameters():
            parameter.requires_grad = False
        layers = self.layers[-last_k_layers:] if last_k_layers else []
        for layer in layers:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        return [parameter for parameter in self.parameters() if parameter.requires_grad]