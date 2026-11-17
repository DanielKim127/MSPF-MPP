from __future__ import annotations
from types import SimpleNamespace
from typing import Dict, Any, List, Tuple
import os
import importlib
import torch
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch.utils.data import Subset
from src.utils.featurizer import smiles_to_data
from .augmentations import make_correlated_graph, random_permutation, permute_graph
from .encoder import SiameseGNN, SimilarityConfig
from .utils import sigma_to_perm, align_accuracy_batch, set_seed


class SimpleSmilesDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        assert "smiles" in df.columns, f"[SimpleSmilesDataset] CSV lack 'smiles': {csv_path}"
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        smi = self.df.loc[idx, "smiles"]
        data = smiles_to_data(smi, y_vec=None, gid=idx, inject_pe=False)
        data.graph_id = int(idx)
        return data


class PairDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        eta: float = 0.1,
        avoid_disconnect: bool = True,
        allow_remove: bool = False,
    ):
        self.base = base_dataset
        self.eta = float(eta)
        self.avoid_disconnect = bool(avoid_disconnect)
        self.allow_remove = bool(allow_remove)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data: Data = self.base[idx]
        
        data_tilde = make_correlated_graph(
            data,
            eta=self.eta,
            avoid_disconnect=self.avoid_disconnect,
            allow_remove=self.allow_remove,
        )
        
        N = int(data_tilde.num_nodes)
        perm = random_permutation(N, device=data_tilde.edge_index.device)  
        data_tilde_perm, perm_used = permute_graph(data_tilde, perm)       

        
        inv = torch.empty_like(perm_used)
        inv[perm_used] = torch.arange(N, device=perm_used.device)          
        perm_gt = inv  

        
        sample = {
            "data": data,
            "data_tilde": data_tilde_perm,
            "perm": perm_gt,   
        }
        return sample


def pair_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    data_list = [b["data"] for b in batch]
    tilde_list = [b["data_tilde"] for b in batch]
    perm_list = [b["perm"] for b in batch]

    pyg_batch = Batch.from_data_list(data_list)
    pyg_tilde = Batch.from_data_list(tilde_list)
    return {
        "data": pyg_batch,
        "data_tilde": pyg_tilde,
        "perm": perm_list,  
    }


def build_pair_loaders(
    name: str,
    eta: float,
    avoid_disconnect: bool,
    allow_remove: bool,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    csv_path: str | None = None,
    val_ratio: float = 0.05,
    split_seed: int = 42,
    pcqm_root: str | None = None,
) -> Tuple[DataLoader, DataLoader]:
    name_l = name.lower()
  
    if name_l in {"pcqm4m", "pcqm4mv2"} and not csv_path:
        try:
            try:
                from ogb.lsc import PCQM4Mv2Dataset as _PCQM
            except Exception:
                from ogb.lsc import PCQM4MDataset as _PCQM
        except Exception as e:
            raise ImportError("install ogb") from e

        
        root = pcqm_root
        if root is None:
            root = os.environ.get("PCQM4M_ROOT", os.path.join("data", "pcqm4m"))
        try:
            pcqm = _PCQM(root=root, only_smiles=True)
        except TypeError:
            pcqm = _PCQM(root=root)

        if hasattr(pcqm, "get_idx_split"):
            split = pcqm.get_idx_split()

            idx_train = split.get("train", None)
            idx_val = split.get("valid", None)
            if idx_val is None:
                idx_val = split.get("val", None)

            if idx_train is None:
                idx_train = list(range(len(pcqm)))
            else:
                idx_train = np.asarray(idx_train).tolist()

            if (idx_val is None) or (len(idx_val) == 0):
                val_n = max(1, int(len(idx_train) * 0.01))
                idx_val = idx_train[:val_n]
                idx_train = idx_train[val_n:]
            else:
                idx_val = np.asarray(idx_val).tolist()
        else:
            idx_train = list(range(len(pcqm)))
            val_n = max(1, int(len(idx_train) * 0.01))
            idx_val = idx_train[:val_n]
            idx_train = idx_train[val_n:]

        _s2g = None
        try:
            ogb_mol = importlib.import_module("ogb.utils.mol")
            _s2g = ogb_mol.smiles2graph
        except Exception:
            _s2g = None

        from torch_geometric.data import Data as _Data

        class _PCQMAsData(torch.utils.data.Dataset):
            def __init__(self, smiles_list, offset=0):
                self.smiles, self.offset = smiles_list, offset

            def __len__(self):
                return len(self.smiles)

            def __getitem__(self, i):
                if _s2g is None:
                    raise RuntimeError("ogb unavailable")

                tries = 3
                j = int(i)
                for _ in range(tries):
                    s = self.smiles[j]
                    try:
                        g = _s2g(s)
                        if g is None or ("node_feat" not in g) or g["node_feat"] is None:
                            raise ValueError("invalid smiles2graph result")
                        x = torch.tensor(g["node_feat"], dtype=torch.float)
                        ei = torch.tensor(g["edge_index"], dtype=torch.long)
                        ea = torch.tensor(g["edge_feat"], dtype=torch.float)
                        if ei.dim() == 2 and ei.size(0) != 2:  
                            ei = ei.t().contiguous()
                        return _Data(x=x, edge_index=ei, edge_attr=ea, y=torch.zeros(1, 1))
                    except Exception:
                        
                        j = (j + 1) % len(self.smiles)

                g = _s2g("C")
                x = torch.tensor(g["node_feat"], dtype=torch.float)
                ei = torch.tensor(g["edge_index"], dtype=torch.long)
                ea = torch.tensor(g["edge_feat"], dtype=torch.float)
                if ei.dim() == 2 and ei.size(0) != 2:
                    ei = ei.t().contiguous()
                return _Data(x=x, edge_index=ei, edge_attr=ea, y=torch.zeros(1, 1))

        def _extract_smiles_list(ds, index_list):
            if hasattr(ds, "smiles"):
                arr = ds.smiles
                return [arr[i] for i in index_list]
            if hasattr(ds, "smiles_list"):
                arr = ds.smiles_list
                return [arr[i] for i in index_list]
            
            res = []
            for i in index_list:
                item = ds[i]
                if isinstance(item, str):
                    s = item
                elif isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], str):
                    s = item[0]
                elif isinstance(item, dict) and ("smiles" in item):
                    s = item["smiles"]
                elif hasattr(item, "smiles"):
                    s = item.smiles
                else:
                    raise RuntimeError("SMILES cant be extracted")
                res.append(s)
            return res

        smiles_train = _extract_smiles_list(pcqm, idx_train)
        smiles_val = _extract_smiles_list(pcqm, idx_val)
        base_train = _PCQMAsData(smiles_train)
        base_val = _PCQMAsData(smiles_val, offset=len(smiles_train))
        ds_train = PairDataset(base_train, eta=eta, avoid_disconnect=avoid_disconnect, allow_remove=allow_remove)
        ds_val = PairDataset(base_val, eta=eta, avoid_disconnect=avoid_disconnect, allow_remove=allow_remove)
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory, collate_fn=pair_collate)
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory, collate_fn=pair_collate)
        return train_loader, val_loader
    
    if csv_path or name_l in {"csv", "pcqm4m", "pcqm4mv2", "freesolv", "esol", "hiv", "bbbp", "bace", "tox21", "toxcast", "sider", "clintox"}:
        if not csv_path:
            raise ValueError(f"{name} pretraining needs smiles")

        base = SimpleSmilesDataset(csv_path=csv_path)

        N = len(base)
        idx = np.arange(N)
        rng = np.random.RandomState(int(split_seed))
        rng.shuffle(idx)

        val_n = max(1, int(float(val_ratio) * N)) if float(val_ratio) > 0 else 0
        if val_n > 0:
            val_idx = idx[:val_n].tolist()
            train_idx = idx[val_n:].tolist()
            base_train = Subset(base, train_idx)
            base_val   = Subset(base, val_idx)
        else:
            base_train = base
            base_val   = base  

        ds_train = PairDataset(base_train, eta=eta, avoid_disconnect=avoid_disconnect, allow_remove=allow_remove)
        ds_val   = PairDataset(base_val,   eta=eta, avoid_disconnect=avoid_disconnect, allow_remove=allow_remove)

        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory, collate_fn=pair_collate)
        val_loader   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory, collate_fn=pair_collate)
        return train_loader, val_loader

    raise ValueError(f"Unsupported dataset")


def build_pretrain_system(args, device: torch.device):
    if hasattr(args, "seed"):
        set_seed(args.seed)

    train_loader, val_loader = build_pair_loaders(
        name=args.dataset,
        eta=float(args.eta),
        avoid_disconnect=bool(getattr(args, "avoid_disconnect", True)),
        allow_remove=False,  
        batch_size=int(args.batch_size),
        num_workers=int(getattr(args, "num_workers", 0)),
        pin_memory=True,
        csv_path=getattr(args, "csv_path", None),
        val_ratio=float(getattr(args, "val_ratio", 0.05)),
        split_seed=int(getattr(args, "split_seed", 42)),
        pcqm_root=getattr(args, "pcqm_root", None),
    )

    sim_cfg = SimilarityConfig(kind=getattr(args, "sim", "dot"), temperature=float(getattr(args, "temperature", 1.0)))
    
    sample: Dict[str, Any] = next(iter(train_loader))
    d0: Data = sample["data"].to(device)
    x_dim = int(d0.x.size(-1)) if (hasattr(d0, "x") and d0.x is not None) else 1
    if hasattr(d0, "edge_attr") and d0.edge_attr is not None:
        
        e_dim = 1 if d0.edge_attr.dim() == 1 else int(d0.edge_attr.size(-1))
    else:
        e_dim = 0

    model = SiameseGNN(
        in_dim=x_dim,
        edge_dim=e_dim,
        hidden=int(getattr(args, "hidden", 256)),
        layers=int(getattr(args, "layers", 5)),
        dropout=float(getattr(args, "dropout", 0.1)),
        sim=sim_cfg,
        use_edge=True,
        normalize_emb=bool(getattr(args, "normalize_emb", False)),
    ).to(device)

    
    lr = float(getattr(args, "lr", 2e-4))
    wd = float(getattr(args, "weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(getattr(args, "epochs", 100))
    
    if getattr(args, "scheduler", "cosine") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None

    def criterion(outputs: Dict[str, Tensor], batch: Dict[str, Any]) -> Tensor:
        
        dev = outputs["X"].device  
        perm_on_dev = [p.to(dev) for p in batch["perm"]]
        return model.align_loss(outputs["Sigma_list"], perm_on_dev)

    @torch.no_grad()
    def infer_perm(m: SiameseGNN, batch: Dict[str, Any]) -> List[Tensor]:
        m.eval()
        outs = m(batch)
        return sigma_to_perm(outs["Sigma_list"], maximize=True)

    @torch.no_grad()
    def eval_align(pred_list: List[Tensor], gt_list: List[Tensor]) -> Tuple[int, int]:
        return align_accuracy_batch(pred_list, gt_list)

    system = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        infer_perm=infer_perm,
        eval_align=eval_align,
    )
    return system
    
def build_encoder_and_dataset_for_export(args, device: torch.device):
    use_csv = getattr(args, "csv_path", None)
    if use_csv is None:
        raise ValueError("--csv_path is required for GMPE embedding export")
    dataset = SimpleSmilesDataset(csv_path=use_csv)

    d0: Data = dataset[0]
    x_dim = int(d0.x.size(-1)) if (hasattr(d0, "x") and d0.x is not None) else 1
    if hasattr(d0, "edge_attr") and d0.edge_attr is not None:
        e_dim = 1 if d0.edge_attr.dim() == 1 else int(d0.edge_attr.size(-1))
    else:
        e_dim = 0

    sim_cfg = SimilarityConfig(kind=getattr(args, "sim", "dot"), temperature=float(getattr(args, "temperature", 1.0)))
    encoder = SiameseGNN(
        in_dim=x_dim,
        edge_dim=e_dim,
        hidden=int(getattr(args, "hidden", getattr(args, "pe_dim", 256))),
        layers=int(getattr(args, "layers", 5)),
        dropout=float(getattr(args, "dropout", 0.1)),
        sim=sim_cfg,
        use_edge=True,
        normalize_emb=bool(getattr(args, "normalize", False)),  
    ).to(device)

    return encoder, dataset