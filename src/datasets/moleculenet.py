import os
from typing import Callable, List, Optional, Dict, Any
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data

DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "BBBP":     dict(task_type="cls", num_tasks=1,  label_cols=["p_np"]),                   
    "BACE":     dict(task_type="cls", num_tasks=1,  label_cols=["Class"]),
    "HIV":      dict(task_type="cls", num_tasks=1,  label_cols=["HIV_active"]),
    "ClinTox":  dict(task_type="cls", num_tasks=2,  label_cols=['FDA_APPROVED','CT_TOX']), 
    "SIDER":    dict(task_type="cls", num_tasks=27, label_cols=None),                      
    "Tox21":    dict(task_type="cls", num_tasks=12, label_cols=["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
                                                                "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma","SR-ARE",
                                                                "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]),
    "ToxCast":  dict(task_type="cls", num_tasks=None, label_cols=None),
    "MUV":      dict(task_type="cls", num_tasks=None, label_cols=["MUV-644"]),
    "ESOL":     dict(task_type="reg", num_tasks=1,  label_cols=["measured log solubility in mols per litre"]),
    "FreeSolv": dict(task_type="reg", num_tasks=1,  label_cols=['expt']),
    "Lipo": dict(task_type="reg", num_tasks=1, label_cols=['exp']),  
}


def _infer_label_columns(df: pd.DataFrame, name: str, spec: Dict[str, Any]) -> List[str]:
    if spec.get("label_cols"):
        return spec["label_cols"]

    cols = list(df.columns)

    if name in ["SIDER"]:
        candidates = [c for c in cols if c.lower() not in ["smiles", "drug", "drugbank id", "name"]]
        return candidates

    if name in ["Tox21", "ToxCast", "MUV"]:
        candidates = [c for c in cols if c.lower() not in ["smiles", "mol_id", "id", "cid"]]
        return candidates

    if name == "ClinTox":
        cand = [c for c in cols if c.upper() in ["CT_TOX", "FDA_APPROVED"]]
        return cand if cand else [c for c in cols if c.lower() not in ["smiles", "id"]]

    numeric_cols = [c for c in cols if c.lower() != "smiles" and pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols


class MoleculeNetDataset(InMemoryDataset):
    def __init__(
        self,
        csv_path: str,
        name: str,
        featurizer: Callable[[str], Optional[Data]],
        pe_builder: Optional[Callable[[Data], Data]] = None,  
        transform=None,
        pre_transform=None,
        **kwargs,
    ):
        self.csv_path = csv_path
        self.name = name
        self.spec = DATASET_SPECS[name]
        self.featurizer = featurizer
        self.pe_builder = pe_builder
        super().__init__(root=os.path.dirname(csv_path), transform=transform,
                         pre_transform=pre_transform)

        
        self.data, self.slices, meta = torch.load(self.processed_paths[0])
        self.label_cols: List[str] = meta["label_cols"]
        self.task_type: str = meta["task_type"]
        self.num_tasks: int = meta["num_tasks"]

    @property
    def raw_file_names(self) -> List[str]:
        return [os.path.basename(self.csv_path)]

    @property
    def processed_file_names(self) -> str:
        return f"pyg_{self.name}.pt"

    def download(self):
        
        pass

    def process(self):
        df = pd.read_csv(self.csv_path)
        
        if "smiles" not in df.columns:
            
            for alt in ["SMILES", "Smiles", "mol", "molecule", "canonical_smiles"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "smiles"})
                    break
        assert "smiles" in df.columns, "must have smiles"

        label_cols = _infer_label_columns(df, self.name, self.spec)
        num_tasks = self.spec["num_tasks"] or len(label_cols)

        data_list: List[Data] = []
        for _, row in df.iterrows():
            smi = str(row["smiles"])
            try:
                g = self.featurizer(smi)  
            except Exception:
                continue
            if g is None:
                continue

            vals = []
            for c in label_cols:
                v = pd.to_numeric(row.get(c, float('nan')), errors='coerce')
                vals.append(v)
            y = torch.tensor(vals, dtype=torch.float32).view(1, -1)

            if torch.isnan(y).all():
                continue

            g.y = y
            g.y_mask = ~torch.isnan(y)  
            g.smiles = smi

            if self.pe_builder is not None and getattr(g, "pe", None) is None:
                g = self.pe_builder(g)

            data_list.append(g)

        assert len(data_list) > 0, f"check featurizer。"

        data, slices = self.collate(data_list)
        meta = dict(label_cols=label_cols, task_type=self.spec["task_type"], num_tasks=num_tasks)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices, meta), self.processed_paths[0])