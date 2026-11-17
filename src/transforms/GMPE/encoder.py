from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv

def mlp(channels, act=nn.GELU, bn=False, dropout=0.0):
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Linear(channels[i], channels[i + 1]))
        if i < len(channels) - 2:
            if bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class GINBackbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden: int = 256,
        layers: int = 5,
        dropout: float = 0.1,
        use_edge: bool = True,
    ):
        super().__init__()
        self.use_edge = use_edge and (edge_dim is not None) and (edge_dim > 0)

        self.node_in = nn.Linear(in_dim, hidden)
        self.edge_in = nn.Linear(edge_dim, hidden) if self.use_edge else None

        convs = []
        norms = []
        for _ in range(layers):
            nn_mlp = mlp([hidden, hidden, hidden], act=nn.GELU, bn=True, dropout=dropout)
            convs.append(GINConv(nn_mlp, train_eps=True))
            norms.append(nn.BatchNorm1d(hidden))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(dropout)

        self.out_proj = nn.Identity()  

    def forward(self, data: Data) -> Tensor:
        x, ei = data.x, data.edge_index

        if x.size(-1) != self.node_in.in_features:
            raise RuntimeError(
                f"[GINBackbone] node feature dim mismatch: got {x.size(-1)}, "
                f"expected {self.node_in.in_features}. "
                f"Hint: construct GINBackbone with in_dim=<dataset.x_dim>."
            )
        x = x.to(self.node_in.weight.dtype)
        
        if x.device != self.node_in.weight.device:
            x = x.to(self.node_in.weight.device)

        if ei.device != x.device:
            ei = ei.to(x.device)

        x = self.node_in(x)

        if self.use_edge:
            e_attr = data.edge_attr
            if e_attr.dim() == 1:
                e_attr = e_attr.unsqueeze(-1)
            if e_attr.size(-1) != self.edge_in.in_features:
                raise RuntimeError(
                    f"[GINBackbone] edge feature dim mismatch: got {e_attr.size(-1)}, "
                    f"expected {self.edge_in.in_features}. "
                    f"Hint: construct GINBackbone with edge_dim=<dataset.edge_dim>."
                )
            e_attr = e_attr.to(self.edge_in.weight.dtype)
            
            if e_attr.device != self.edge_in.weight.device:
                e_attr = e_attr.to(self.edge_in.weight.device)
        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, ei)  
            x = bn(x)
            x = F.gelu(x)
            x = self.dropout(x)
        return self.out_proj(x)

    def flexible_load(self, state: Dict[str, Tensor], skip_inputs: bool = True, verbose: bool = True):
        own = self.state_dict()
        filtered = {}
        for k, v in state.items():
            
            k_clean = k.split("backbone.", 1)[-1] if k.startswith("backbone.") else k
            if skip_inputs and (k_clean.startswith("node_in.") or k_clean.startswith("edge_in.")):
                continue
            if k_clean in own and own[k_clean].shape == v.shape:
                filtered[k_clean] = v
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        if verbose:
            print(f"[GINBackbone.flexible_load] loaded: {len(filtered)} tensors")
            if missing:   print("missing:", missing)
            if unexpected: print("unexpected:", unexpected)
        return missing, unexpected


class SimilarityConfig:
    kind: Literal["dot", "cos"] = "dot"
    temperature: float = 1.0


class SiameseGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden: int = 256,
        layers: int = 5,
        dropout: float = 0.1,
        sim: SimilarityConfig = SimilarityConfig(),
        use_edge: bool = True,
        normalize_emb: bool = False,
    ):
        super().__init__()
        self.backbone = GINBackbone(in_dim, edge_dim, hidden, layers, dropout, use_edge)
        self.sim_cfg = sim
        self.normalize_emb = normalize_emb
        self.hidden = hidden

    @torch.no_grad()
    def encode_single_graph(self, data: Data) -> Tensor:
        X = self.backbone(data)
        if self.normalize_emb:
            X = X - X.mean(dim=0, keepdim=True)
            X = F.normalize(X, dim=1)
        return X

    def forward(self, batch_pair: Dict[str, Any]) -> Dict[str, Tensor]:
        G: Batch = batch_pair["data"]
        Gt: Batch = batch_pair["data_tilde"]
        
        X  = self.backbone(G)   
        Xt = self.backbone(Gt)  

        if self.normalize_emb:
            X  = X - X.mean(dim=0, keepdim=True);  X  = F.normalize(X, dim=1)
            Xt = Xt - Xt.mean(dim=0, keepdim=True); Xt = F.normalize(Xt, dim=1)

        sigmas = []
        xs, xst = [], []
        for gid in G.batch.unique().tolist():
            mask_i = (G.batch == gid)
            xs.append(X[mask_i])
        for gid in Gt.batch.unique().tolist():
            mask_j = (Gt.batch == gid)
            xst.append(Xt[mask_j])

        assert len(xs) == len(xst)

        for Xi, Xj in zip(xs, xst):
            Sigma = self.build_similarity(Xi, Xj)  
            sigmas.append(Sigma)

        return {"Sigma_list": sigmas, "X": X, "X_tilde": Xt}


    def build_similarity(self, X: Tensor, Xt: Tensor) -> Tensor:
        if self.sim_cfg.kind == "dot":
            S = (X @ Xt.t()) / max(1e-6, self.sim_cfg.temperature)
        elif self.sim_cfg.kind == "cos":
            Xn  = F.normalize(X,  dim=1)
            Xtn = F.normalize(Xt, dim=1)
            S = (Xn @ Xtn.t()) / max(1e-6, self.sim_cfg.temperature)

        return S

    @staticmethod
    def align_loss(
        Sigma_list: list[Tensor],
        perm_list: list[Tensor],
        reduction: str = "mean",
    ) -> Tensor:
        
        assert len(Sigma_list) == len(perm_list)
        losses = []
        for S, perm in zip(Sigma_list, perm_list):
            row_loss = F.cross_entropy(S, perm, reduction="mean")

            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)

            col_loss = F.cross_entropy(S.transpose(0, 1), inv_perm, reduction="mean")
            losses.append(0.5 * (row_loss + col_loss))

        stacked = torch.stack(losses)
        return stacked.mean() if reduction == "mean" else stacked.sum()

    def flexible_load(self, state: Dict[str, Tensor], skip_inputs: bool = True, verbose: bool = True):
        return self.backbone.flexible_load(state, skip_inputs=skip_inputs, verbose=verbose)

    def param_groups(self, lr_backbone: float, lr_inputs: Optional[float] = None):
        inputs, rest = [], []
        for n, p in self.backbone.named_parameters():
            if n.startswith(("node_in.", "edge_in.")):
                inputs.append(p)
            else:
                rest.append(p)
        return [
            {"params": inputs, "lr": lr_inputs or lr_backbone},
            {"params": rest,   "lr": lr_backbone},
        ]