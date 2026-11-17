import torch


def mask_atom_attributes(data, ratio=0.15):
    masked = data.clone()
    indices = []
    for gid in data.batch.unique():
        atoms = (data.batch == gid).nonzero(as_tuple=False).flatten()
        count = max(1, round(atoms.numel() * ratio))
        indices.append(atoms[torch.randperm(atoms.numel(), device=atoms.device)[:count]])
    index = torch.cat(indices)
    targets = {'atomic': data.x[index, :118].argmax(-1),
               'hybrid': data.x[index, 118:123].argmax(-1)}
    masked.x[index] = 0
    return masked, index, targets