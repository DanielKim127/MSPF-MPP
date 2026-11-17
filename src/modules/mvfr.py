from torch import nn
import torch

class MVFR(nn.Module):
    def __init__(self, view_dims, hidden=128, heads=4, dropout=0.1):
        super().__init__()
        self.projections = nn.ModuleList([nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                                         nn.Linear(hidden, hidden)) for dim in view_dims])
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, views):
        tokens = torch.stack([project(view) for project, view in zip(self.projections, views)], 1)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + self.drop(attended))
        tokens = self.norm2(tokens + self.drop(self.ffn(tokens)))
        return self.output(tokens.mean(1))