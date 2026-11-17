import torch
from torch import nn
from src.modules.kl_heads import DiagGaussHead

class VariationalReconstruction(nn.Module):
    def __init__(self, dim, latent_dim=128, beta=0.01):
        super().__init__()
        self.posterior2d, self.posterior3d = DiagGaussHead(dim, latent_dim), DiagGaussHead(dim, latent_dim)
        self.decode2d = nn.Sequential(nn.Linear(latent_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.decode3d = nn.Sequential(nn.Linear(latent_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.beta = beta

    def sample(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def prior_kl(mu, logvar):
        return 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()

    def forward(self, h2d, h3d):
        m2, v2 = self.posterior2d(h2d)
        m3, v3 = self.posterior3d(h3d)
        rec = ((self.decode2d(self.sample(m2, v2)) - h3d.detach()).square().sum(-1).mean()
               + (self.decode3d(self.sample(m3, v3)) - h2d.detach()).square().sum(-1).mean()) / 2
        kl = (self.prior_kl(m2, v2) + self.prior_kl(m3, v3)) / 2
        return rec + self.beta * kl