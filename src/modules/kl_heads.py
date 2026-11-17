from __future__ import annotations
import torch
import torch.nn as nn
from typing import Literal, Tuple

__all__ = [
    "DiagGaussHead",
    "diag_gauss_kl",
    "bernoulli_kl_from_logits",
    "gaussian_pred_kl",
]


class DiagGaussHead(nn.Module):
    
    def __init__(self, in_dim: int, z_dim: int, init_logvar_bias: float = -2.0):
        super().__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.lv = nn.Linear(in_dim, z_dim)
        nn.init.zeros_(self.mu.bias)
        
        with torch.no_grad():
            if self.lv.bias is not None:
                self.lv.bias.fill_(float(init_logvar_bias))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(h)
        lv = self.lv(h).clamp(min=-8.0, max=8.0)  
        return mu, lv


def diag_gauss_kl(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    reduce: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    
    kl = 0.5 * (
        (var_q / var_p)
        + (mu_p - mu_q).pow(2) / var_p
        - 1.0
        + (logvar_p - logvar_q)
    ).sum(dim=-1)  

    if reduce == "mean":
        return kl.mean()
    if reduce == "sum":
        return kl.sum()
    return kl  


def bernoulli_kl_from_logits(
    logits_q: torch.Tensor,
    logits_p: torch.Tensor,
    T: float = 1.0,
    reduce: Literal["mean", "sum", "none"] = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    
    q = torch.sigmoid(logits_q / T).clamp(eps, 1 - eps)
    p = torch.sigmoid(logits_p / T).clamp(eps, 1 - eps)
    kl = q * (q.log() - p.log()) + (1 - q) * ((1 - q).log() - (1 - p).log())
    kl = kl.sum(dim=-1)  

    if reduce == "mean":
        return kl.mean()
    if reduce == "sum":
        return kl.sum()
    return kl


def gaussian_pred_kl(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    symmetric: bool = True,
    reduce: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    
    
    if mu_q.dim() == 1:
        mu_q = mu_q.unsqueeze(-1)
        mu_p = mu_p.unsqueeze(-1)
        logvar_q = logvar_q.unsqueeze(-1)
        logvar_p = logvar_p.unsqueeze(-1)

    var_q = logvar_q.exp()
    var_p = logvar_p.exp()

    kl_qp = 0.5 * (
        (var_q / var_p)
        + (mu_p - mu_q).pow(2) / var_p
        - 1.0
        + (logvar_p - logvar_q)
    ).sum(dim=-1)  

    if symmetric:
        kl_pq = 0.5 * (
            (var_p / var_q)
            + (mu_q - mu_p).pow(2) / var_q
            - 1.0
            + (logvar_q - logvar_p)
        ).sum(dim=-1)
        kl = kl_qp + kl_pq
    else:
        kl = kl_qp

    if reduce == "mean":
        return kl.mean()
    if reduce == "sum":
        return kl.sum()
    return kl