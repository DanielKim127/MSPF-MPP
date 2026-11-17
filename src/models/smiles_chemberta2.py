import os
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModel, AutoConfig

class SmilesChemBERTa2Encoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = None,
        proj_out_dim: Optional[int] = None,
        pooling: str = "cls",     
        max_len: int = 256,
        fp16: bool = False,
        dropout: float = 0.0,
        encoder_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path or os.getenv(
            "CHEMBERTA2_MODEL", "DeepChem/ChemBERTa-77M-MLM"
        )
        self.pooling = pooling.lower()
        self.max_len = max_len
        self.fp16 = fp16

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        if encoder_config is None:
            self.encoder = AutoModel.from_pretrained(self.model_name_or_path)
        else:
            config_dict = dict(encoder_config)
            model_type = config_dict.pop('model_type')
            self.encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **config_dict))
        d_model = self.encoder.config.hidden_size

        self.do_proj = proj_out_dim is not None
        if self.do_proj:
            self.proj = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, proj_out_dim)
            )
        else:
            self.proj = nn.Identity()

    def encode_smiles(self, smiles: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        batch = self.tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        if self.fp16:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                out = self.encoder(**batch)
        else:
            out = self.encoder(**batch)

        last_hidden = out.last_hidden_state  
        if self.pooling == "mean":
            mask = batch["attention_mask"].unsqueeze(-1)  
            emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        else:  
            
            emb = last_hidden[:, 0, :]  

        emb = self.proj(emb)  
        return emb
    
    @torch.no_grad()
    def extract_smiles(self, smiles, device=None):
        return self.encode_smiles(smiles, device=device)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def freeze_all(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        if hasattr(self, "proj"):
            for p in self.proj.parameters():
                p.requires_grad = True

    def set_finetune(self, last_k_layers: int = 0, unfreeze_pooler_proj: bool = True):
        for p in self.encoder.parameters():
            p.requires_grad = False

        if last_k_layers == -1:
            
            for p in self.encoder.parameters():
                p.requires_grad = True
        elif last_k_layers > 0:
            import re
            L = getattr(self.encoder.config, "num_hidden_layers", 12)
            keep = set(range(max(0, L - last_k_layers), L))
            for name, p in self.encoder.named_parameters():
                m = re.search(r"encoder\.layer\.(\d+)\.", name)
                if m and (int(m.group(1)) in keep):
                    p.requires_grad = True

        if unfreeze_pooler_proj:
            if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
                for p in self.encoder.pooler.parameters():
                    p.requires_grad = (last_k_layers != 0)
            if hasattr(self, "proj"):
                for p in self.proj.parameters():
                    p.requires_grad = True

        return self.trainable_parameters()

    def forward(self, data: Dict[str, Any]):
        smiles_list = data.get("smiles", None)
        if smiles_list is None:
            raise ValueError("Expected 'smiles' in data but not found.")
        device = next(self.parameters()).device
        data.smiles_emb = self.encode_smiles(smiles_list, device=device)
        return data.smiles_emb