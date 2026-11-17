import torch
import torch.distributed as dist
import torch.nn.functional as F


def observed_labels(target, task_type):
    valid = ~torch.isnan(target)
    if task_type == 'cls':
        valid &= target >= 0
    elif task_type != 'reg':
        raise ValueError(f'Unknown task type: {task_type}')
    if (valid & ~torch.isfinite(target)).any():
        raise ValueError('labels must be finite')
    return valid


def training_class_weights(records, train_indices, num_tasks):
    counts = torch.zeros((num_tasks, 2), dtype=torch.long)
    for index in train_indices:
        target = torch.as_tensor(records[index]['y']).detach().cpu().reshape(-1)
        valid = observed_labels(target, 'cls')
        counts[:, 0] += (valid & (target == 0)).long()
        counts[:, 1] += (valid & (target == 1)).long()
    weights = torch.ones((num_tasks, 2), dtype=torch.float64)
    both = (counts > 0).all(dim=1)
    balanced_counts = counts[both].to(weights.dtype)
    weights[both] = balanced_counts.sum(dim=1, keepdim=True) / (2 * balanced_counts)
    info = {
        'name': 'masked_balanced_bce',
        'weight_source': 'training_split_only',
        'normalization': 'mean_over_observed_entries',
        'class_order': ['negative', 'positive'],
        'negative_counts': counts[:, 0].tolist(),
        'positive_counts': counts[:, 1].tolist(),
        'class_weights': weights.tolist(),
        'tasks_without_both_classes': (~both).nonzero(as_tuple=True)[0].tolist(),
        'degenerate_task_policy': 'unit_weights',
    }
    return weights.float(), info


def masked_task_loss(prediction, target, task_type, class_weights=None,
                     distributed_normalization=False):
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if target.dim() == 1:
        target = target.unsqueeze(1)
    valid = observed_labels(target, task_type)
    if task_type == 'cls':
        entry_weights = None
        if class_weights is not None:
            weights = torch.as_tensor(class_weights, device=prediction.device,
                                      dtype=prediction.dtype)
            
            task_ids = valid.nonzero(as_tuple=True)[1]
            entry_weights = weights[task_ids, target[valid].long()]
        total = F.binary_cross_entropy_with_logits(
            prediction[valid], target[valid], weight=entry_weights, reduction='sum')
    else:
        total = F.mse_loss(prediction[valid], target[valid], reduction='sum')
    count = valid.sum().to(dtype=prediction.dtype)
    if distributed_normalization:
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        total = total * dist.get_world_size()
    
    return total / count.clamp_min(1)