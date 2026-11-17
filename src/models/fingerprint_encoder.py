from torch import nn

class FingerprintEncoder(nn.Module):
    def __init__(self, in_dim=1048, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, hidden))

    def forward(self, fingerprint):
        return self.net(fingerprint)