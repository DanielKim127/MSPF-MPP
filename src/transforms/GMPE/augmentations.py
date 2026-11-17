from __future__ import annotations
import random
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, to_undirected


def _canonical_undirected(edge_index: Tensor, num_nodes: int) -> Tensor:
    ei = edge_index
    mask = ei[0] != ei[1]
    ei = ei[:, mask]
    ei = to_undirected(ei, num_nodes=num_nodes)
    ei, _ = coalesce(ei, None, num_nodes=num_nodes)
    return ei

def _edge_set(edge_index: Tensor) -> set[Tuple[int, int]]:
    return set((int(u), int(v)) for u, v in edge_index.t().tolist())

def _candidate_non_edges(num_nodes: int, existing: set[Tuple[int, int]], max_trials: int = 100000) -> list[Tuple[int, int]]:
    cand = []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if (u, v) not in existing and (v, u) not in existing:
                cand.append((u, v))
    random.shuffle(cand)
    return cand

def _check_connected(num_nodes: int, edge_index: Tensor) -> bool:
    if num_nodes == 0:
        return True
    ei = to_undirected(edge_index, num_nodes=num_nodes)
    adj = [[] for _ in range(num_nodes)]
    for u, v in ei.t().tolist():
        adj[u].append(v)
        adj[v].append(u)
    seen = [False] * num_nodes
    q = [0]
    seen[0] = True
    for u in q:
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return all(seen)

@torch.no_grad()
def make_correlated_graph(
    data: Data,
    eta: float = 0.1,
    avoid_disconnect: bool = True,
    allow_remove: bool = False,
    max_add_ratio: Optional[float] = None,
) -> Data:
    
    assert 0.0 <= eta < 1.0
    N = int(data.num_nodes)
    device = data.edge_index.device
    x = data.x.clone() if data.x is not None else None
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone() if "edge_attr" in data and data.edge_attr is not None else None
    
    if edge_attr is not None and edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(-1)

    edge_index = _canonical_undirected(edge_index, N)
    E = edge_index.size(1)
    existing = _edge_set(edge_index)

    n_ops = max(1, int(round(eta * E)))
    n_add_target = n_ops if not allow_remove else n_ops // 2
    n_rem_target = 0 if not allow_remove else n_ops - n_add_target
    if max_add_ratio is not None:
        n_add_target = min(n_add_target, int(round(max_add_ratio * E)))

    if allow_remove and n_rem_target > 0 and E > 0:
        undirected_edges = list({(min(u, v), max(u, v)) for (u, v) in existing})
        random.shuffle(undirected_edges)
        removed = 0
        keep_mask = torch.ones(E, dtype=torch.bool, device=device)
        
        idx_map = {}
        for idx, (u, v) in enumerate(edge_index.t().tolist()):
            a, b = (u, v) if u < v else (v, u)
            idx_map.setdefault((a, b), []).append(idx)

        for (u, v) in undirected_edges:
            if removed >= n_rem_target:
                break
            
            for idx in idx_map[(u, v)]:
                keep_mask[idx] = False
            ei_tmp = edge_index[:, keep_mask]
            if avoid_disconnect and not _check_connected(N, ei_tmp):
                
                for idx in idx_map[(u, v)]:
                    keep_mask[idx] = True
                continue
            
            removed += 1
        edge_index = edge_index[:, keep_mask]
        if edge_attr is not None:
            edge_attr = edge_attr[keep_mask]

        edge_index = _canonical_undirected(edge_index, N)
        existing = _edge_set(edge_index)
        E = edge_index.size(1)

    if n_add_target > 0:
        cand_non_edges = _candidate_non_edges(N, existing)
        added = 0
        add_ei = []
        add_attr = []
        
        default_eattr = None
        if edge_attr is not None:
            default_eattr = torch.zeros(edge_attr.size(-1), device=device, dtype=edge_attr.dtype)

        for (u, v) in cand_non_edges:
            if added >= n_add_target:
                break
            
            add_ei.append([u, v])
            add_ei.append([v, u])  
            if default_eattr is not None:
                add_attr.append(default_eattr)
                add_attr.append(default_eattr)
            added += 1

        if added > 0:
            add_ei = torch.tensor(add_ei, device=device, dtype=edge_index.dtype).t().contiguous()
            edge_index = torch.cat([edge_index, add_ei], dim=1)
            if edge_attr is not None:
                add_attr = torch.stack(add_attr, dim=0)
                edge_attr = torch.cat([edge_attr, add_attr], dim=0)
    
    edge_index = _canonical_undirected(edge_index, N)

    out = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    for k in ["batch", "pos"]:
        if hasattr(data, k):
            setattr(out, k, getattr(data, k))
    return out


def random_permutation(n: int, device: Optional[torch.device] = None) -> Tensor:
    perm = torch.randperm(n, device=device)
    return perm


def permute_graph(data: Data, perm: Tensor) -> Tuple[Data, Tensor]:
    N = int(data.num_nodes)
    assert perm.numel() == N, "perm length must equal #nodes"

    x = data.x
    x_perm = x[perm] if x is not None else None

    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(N, device=perm.device)

    ei = data.edge_index
    
    ei_perm = inv[ei]

    edge_attr = data.edge_attr
    edge_attr_perm = edge_attr.clone() if edge_attr is not None else None  

    out = Data(x=x_perm, edge_index=ei_perm, edge_attr=edge_attr_perm)
    for k in ["batch", "pos"]:
        if hasattr(data, k):
            setattr(out, k, getattr(data, k))
    return out, perm