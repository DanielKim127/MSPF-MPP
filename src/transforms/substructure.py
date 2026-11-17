import torch


def paired_substructure_pool(nodes2d, nodes3d, data, seeds_per_molecule=1, hops=2):
    src, dst = data.edge_index
    first, second = [], []
    for gid in data.batch.unique():
        atoms = (data.batch == gid).nonzero(as_tuple=False).flatten()
        seeds = atoms[torch.randperm(atoms.numel(), device=atoms.device)[:seeds_per_molecule]]
        included = torch.zeros(data.num_nodes, dtype=torch.bool, device=atoms.device)
        included[seeds] = True
        for _ in range(hops):
            neighbors = dst[included[src]]
            included[neighbors] = True
        assert (data.batch[included] == gid).all()
        first.append(nodes2d[included].mean(0))
        second.append(nodes3d[included].mean(0))
    return torch.stack(first), torch.stack(second)