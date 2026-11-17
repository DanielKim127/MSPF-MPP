from torch import nn
from torch.nn import functional as F
from src.modules.contrastive import cross_view_nt_xent
from src.modules.variational_reconstruction import VariationalReconstruction
from src.transforms.attr_mask import mask_atom_attributes
from src.transforms.substructure import paired_substructure_pool

class Pretrain2D3D(nn.Module):
    def __init__(self, encoder2d, encoder3d, hidden, temperature=0.1, mask_ratio=0.15,
                 latent_dim=128, beta=0.01, seeds_per_molecule=1,
                 alpha_global=1.0, alpha_sub=1.0, alpha_rec=1.0, alpha_attr=1.0):
        super().__init__()
        self.encoder2d, self.encoder3d = encoder2d, encoder3d
        self.atom_head, self.hybrid_head = nn.Linear(hidden, 118), nn.Linear(hidden, 5)
        self.vrr = VariationalReconstruction(hidden, latent_dim, beta)
        self.temperature, self.mask_ratio = temperature, mask_ratio
        self.seeds_per_molecule = seeds_per_molecule
        self.weights = dict(global_loss=alpha_global, sub_loss=alpha_sub,
                            rec_loss=alpha_rec, attr_loss=alpha_attr)

    def forward(self, data):
        
        two, three = self.encoder2d.encode(data), self.encoder3d(data)
        sub2, sub3 = paired_substructure_pool(two['node_emb'], three['node_emb'], data,
                                              self.seeds_per_molecule)
        masked, index, targets = mask_atom_attributes(data, self.mask_ratio)
        masked_features = self.encoder2d.encode(masked)['node_emb'][index]
        attr = F.cross_entropy(self.atom_head(masked_features), targets['atomic'])
        attr = attr + F.cross_entropy(self.hybrid_head(masked_features), targets['hybrid'])
        losses = {'global_loss': cross_view_nt_xent(two['graph_emb'], three['graph_emb'], self.temperature),
                  'sub_loss': cross_view_nt_xent(sub2, sub3, self.temperature),
                  'rec_loss': self.vrr(two['graph_emb'], three['graph_emb']), 'attr_loss': attr}
        losses['loss'] = sum(self.weights[key] * value for key, value in losses.items())
        return losses