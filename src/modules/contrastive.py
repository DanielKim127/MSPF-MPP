import torch
from torch.nn import functional as F

def cross_view_nt_xent(first, second, temperature=0.1):
    similarity = F.normalize(first, dim=-1) @ F.normalize(second, dim=-1).T / temperature
    target = torch.arange(first.size(0), device=first.device)
    return (F.cross_entropy(similarity, target) + F.cross_entropy(similarity.T, target)) / 2