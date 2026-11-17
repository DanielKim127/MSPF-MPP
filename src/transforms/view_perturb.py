import torch


def perturb_views(views, noise_std=0.1, noise_fraction=0.5, missing_probability=0.5):
    batch, device = views[0].size(0), views[0].device
    samples = torch.rand(batch, 1, device=device) < noise_fraction
    noisy = [v + torch.randn_like(v) * noise_std * samples for v in views]
    mask = torch.ones(3, dtype=torch.bool, device=device)
    
    if torch.rand((), device=device) < missing_probability:
        mask[torch.randint(3, (), device=device)] = False
    missing = [v * mask[i] for i, v in enumerate(views)]
    return noisy, missing, mask, samples.squeeze(-1)