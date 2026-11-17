import copy
from pathlib import Path
from torch import nn
import torch
from src.models.fingerprint_encoder import FingerprintEncoder
from src.models.smiles_chemberta2 import SmilesChemBERTa2Encoder
from src.modules.mvfr import MVFR
from src.modules.contrastive import cross_view_nt_xent
from src.transforms.view_perturb import perturb_views
from src.utils.multiview import build_dspgnn, path_from_root

class MultiviewPredictor(nn.Module):
    def __init__(self, config, smiles_path=None):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.encoder2d = build_dspgnn(config['dspgnn'])
        smiles_cfg = copy.deepcopy(config['smiles'])
        source = smiles_path or smiles_cfg.get('model_name_or_path')
        if source:
            local_path = path_from_root(Path(source).expanduser())
            
            smiles_cfg['model_name_or_path'] = str(local_path) if local_path.is_dir() else str(source)
        self.encoder1d = SmilesChemBERTa2Encoder(**smiles_cfg)
        self.encoder_fp = FingerprintEncoder(**config['fingerprint'])
        self.max_attention_pairs = 0
        dim2 = config['dspgnn']['hidden'] * (2 if config['dspgnn'].get('readout') == 'set2set' else 1)
        dim1 = config['smiles']['proj_out_dim']
        dimfp = config['fingerprint']['hidden']
        hidden = config['prediction_hidden']
        self.prediction_head = nn.Sequential(nn.Linear(dimfp + dim1 + dim2, hidden), nn.GELU(),
                                             nn.Dropout(config.get('dropout', 0.1)),
                                             nn.Linear(hidden, config['num_tasks']))
        self.mvfr = MVFR([dimfp, dim1, dim2], **config['mvfr'])

    def encode_views(self, data):
        return [self.encoder_fp(data.fp), self.encoder1d.encode_smiles(data.smiles),
                self._encode_graph_view(data)]

    def _encode_graph_view(self, data):
        limit = self.max_attention_pairs
        if not limit or not hasattr(data, 'ptr'):
            return self.encoder2d.encode(data)['graph_emb']
        sizes = (data.ptr[1:] - data.ptr[:-1]).detach().cpu().tolist()
        costs = [n * (n - 1) if n > 1 else 1 for n in sizes]
        if sum(costs) <= limit:
            return self.encoder2d.encode(data)['graph_emb']
        from torch_geometric.data import Batch
        graphs = data.to_data_list()
        outputs, group, pairs = [], [], 0
        for graph, cost in zip(graphs, costs):
            if group and pairs + cost > limit:
                outputs.append(self.encoder2d.encode(Batch.from_data_list(group))['graph_emb'])
                group, pairs = [], 0
            group.append(graph)
            pairs += cost
        if group:
            outputs.append(self.encoder2d.encode(Batch.from_data_list(group))['graph_emb'])
            
        return torch.cat(outputs, dim=0)

    def forward(self, data, regularize=False, perturbation=None, temperature=0.1):
        views = self.encode_views(data)
        result = {'prediction': self.prediction_head(torch.cat(views, dim=-1))}
        if regularize:
            noisy, missing, mask, samples = perturb_views(views, **(perturbation or {}))
            first, second = self.mvfr(noisy), self.mvfr(missing)
            result['mvfr_loss'] = cross_view_nt_xent(first, second, temperature)
            result['view_mask'], result['noise_samples'] = mask, samples
        return result

    def set_finetune(self, last_1d=2, last_2d=2):
        self.encoder1d.set_finetune(last_1d)
        self.encoder2d.set_finetune(last_2d)
        
        pooler = getattr(self.encoder1d.encoder, 'pooler', None)
        if pooler is not None and last_1d >= 0:
            for parameter in pooler.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode:
            if not any(p.requires_grad for p in self.encoder1d.encoder.embeddings.parameters()):
                self.encoder1d.encoder.embeddings.eval()
            for layer in self.encoder1d.encoder.encoder.layer:
                if not any(p.requires_grad for p in layer.parameters()):
                    layer.eval()
            for layer in self.encoder2d.layers:
                if not any(p.requires_grad for p in layer.parameters()):
                    layer.eval()
        return self