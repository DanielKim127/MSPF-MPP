from __future__ import annotations
from typing import List, Tuple, Iterable
import os
import random
import numpy as np
import torch
from torch import Tensor
from scipy.optimize import linear_sum_assignment


def set_seed(seed: int = 42, cuda_deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def hungarian(S: Tensor, maximize: bool = True) -> Tensor:
    assert S.dim() == 2, "S must be a 2D matrix"
    N = S.size(0)
    S_np = S.detach().cpu().to(torch.float64).numpy()
    cost = -S_np if maximize else S_np
    row, col = linear_sum_assignment(cost)
    perm = np.full((N,), -1, dtype=np.int64)
    perm[row] = col
    return torch.from_numpy(perm)


def sigma_to_perm(sigmas: Iterable[Tensor], maximize: bool = True) -> List[Tensor]:
    perms = []
    for S in sigmas:
        perms.append(hungarian(S, maximize=maximize))
    return perms


def align_accuracy(pred_perm: Tensor, gt_perm: Tensor) -> Tuple[int, int]:
    assert pred_perm.dim() == 1 and gt_perm.dim() == 1
    N = gt_perm.size(0)
    hits = int((pred_perm[:N] == gt_perm).sum().item())
    return hits, N


def align_accuracy_batch(pred_list: List[Tensor], gt_list: List[Tensor]) -> Tuple[int, int]: 
    assert len(pred_list) == len(gt_list)
    hits = 0
    total = 0
    for p, g in zip(pred_list, gt_list):
        h, n = align_accuracy(p, g)
        hits += h
        total += n
    return hits, total


def normalize_node_emb(X: Tensor, zero_mean: bool = True, l2: bool = True) -> Tensor:
    if zero_mean:
        X = X - X.mean(dim=0, keepdim=True)
    if l2:
        X = torch.nn.functional.normalize(X, dim=1)
    return X


def split_by_batch(X: Tensor, batch_vec: Tensor) -> List[Tensor]:
    out: List[Tensor] = []
    for gid in batch_vec.unique().tolist():
        mask = (batch_vec == gid)
        out.append(X[mask])
    return out