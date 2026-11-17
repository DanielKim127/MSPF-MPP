import argparse
import copy
import json
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from src.datasets.multiview_moleculenet import MultiviewDataset, scaffold_splits, validate_splits
from src.models.multiview_predictor import MultiviewPredictor
from src.utils.task_loss import masked_task_loss, observed_labels, training_class_weights
from src.utils.multiview import (load_config, path_from_root, seed_all, choose_device,
    write_json, finite_gradients, rng_state, restore_rng, state_digest, file_digest)


def _reduce_sum_count(sums, count, device, distributed):
    if not distributed:
        return sums, count
    keys = sorted(sums)
    packed = torch.tensor([count] + [sums[key] for key in keys],
                          dtype=torch.float64, device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return ({key: packed[index + 1].item() for index, key in enumerate(keys)},
            int(packed[0].item()))


def _sync_cuda(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _peak_allocated_mib(device, distributed):
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    if distributed:
        packed = torch.tensor(peak, dtype=torch.float64, device=device)
        dist.all_reduce(packed, op=dist.ReduceOp.MAX)
        peak = packed.item()
    return peak if device.type == 'cuda' else None


def _gather_predictions(prediction, target, distributed, world_size):
    if not distributed:
        return prediction, target
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, (prediction, target))
    predictions = [item[0] for item in gathered if item is not None]
    targets = [item[1] for item in gathered if item is not None]
    prediction = torch.cat(predictions, dim=0) if predictions else torch.empty((0, 1))
    target = torch.cat(targets, dim=0) if targets else torch.empty((0, 1))
    return prediction, target


def _random_splits(size, seed, fractions=(0.8, 0.1, 0.1)):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randperm(size, generator=generator).tolist()
    ntrain = int(round(size * fractions[0]))
    nval = int(round(size * fractions[1]))
    ntrain = min(max(1, ntrain), size - 2)
    nval = min(max(1, nval), size - ntrain - 1)
    return {'train': ids[:ntrain], 'val': ids[ntrain:ntrain + nval],
            'test': ids[ntrain + nval:]}


def evaluate(model, loader, device, task_type, max_steps=0,
             distributed=False, world_size=1, class_weights=None):
    model.eval()
    total, count, predictions, targets = 0.0, 0, [], []
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            batch = batch.to(device)
            output = model(batch, regularize=False)['prediction']
            loss = masked_task_loss(output, batch.y, task_type, class_weights)
            observed_count = int(observed_labels(batch.y, task_type).sum().item())
            total += loss.item() * observed_count
            count += observed_count
            predictions.append(output.cpu()); targets.append(batch.y.cpu())
            if max_steps and step + 1 >= max_steps:
                break
    prediction = torch.cat(predictions) if predictions else torch.empty((0, 1))
    target = torch.cat(targets) if targets else torch.empty((0, 1))
    sums, count = _reduce_sum_count({'loss': total}, count, device, distributed)
    prediction, target = _gather_predictions(prediction, target, distributed, world_size)
    return sums['loss'] / max(1, count), prediction, target


def metric(prediction, target, task_type):
    import numpy as np
    from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
    p, y = prediction.numpy(), target.numpy()
    values = []
    probabilities = 1 / (1 + np.exp(-p)) if task_type == 'cls' else None
    for column in range(p.shape[1]):
        mask = ~np.isnan(y[:, column])
        if task_type == 'cls':
            mask &= y[:, column] >= 0
        if mask.sum() == 0:
            continue
        if task_type == 'cls':
            if len(np.unique(y[mask, column])) < 2:
                continue
            values.append((roc_auc_score(y[mask, column], probabilities[mask, column]),
                           average_precision_score(y[mask, column], probabilities[mask, column])))
        else:
            values.append(mean_squared_error(y[mask, column], p[mask, column]) ** 0.5)
    if task_type == 'cls':
        return {'AUROC': float(np.mean([v[0] for v in values])) if values else None,
                'AUPRC': float(np.mean([v[1] for v in values])) if values else None}
    return {'RMSE': float(np.mean(values)) if values else None}


def _selection_score(metrics, task_type):
    value = metrics.get('AUROC' if task_type == 'cls' else 'RMSE')
    if value is None:
        return -float('inf')
    return value if task_type == 'cls' else -value


def _save_predictions(path, epoch, prediction, target, val_metrics, test_metrics,
                      test_indices, selection):
    torch.save({
        'epoch': epoch,
        'prediction': prediction,
        'target': target,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'test_indices': test_indices,
        'selection': selection,
    }, path)


def _pretrained_encoder2d_state(pretrain):
    model_state = pretrain['model']
    prefix = 'encoder2d.'
    encoder_state = {
        name[len(prefix):]: value
        for name, value in model_state.items()
        if name.startswith(prefix)
    }
    if not encoder_state:
        raise KeyError('no model or weights')
    return encoder_state


def _downstream_gmpe(dataset, pretrain):
    gmpe = dataset.payload['gmpe']
    if not isinstance(gmpe, dict) or not {'config', 'state'}.issubset(gmpe):
        raise ValueError('GMPE generator required to generate PE')
    if gmpe['config']['hidden'] != dataset.metadata['gmpe_dim']:
        raise ValueError('dimension ERROR')
    pretrained_gmpe = pretrain.get('gmpe')
    if pretrained_gmpe is not None:
        if (gmpe['config'] != pretrained_gmpe.get('config') or
                state_digest(gmpe['state']) != state_digest(pretrained_gmpe['state'])):
            raise ValueError('different generators')
    return gmpe


def _local_chemberta_path():
    model_dir = Path(__file__).resolve().parent / 'weights' / 'chemberta2'
    required = ('config.json', 'pytorch_model.bin', 'tokenizer_config.json',
                'vocab.json', 'merges.txt')
    missing = [name for name in required if not (model_dir / name).is_file()]
    
    return str(model_dir)


def run(config, resume=None):
    config = copy.deepcopy(config)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    is_main = rank == 0
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError('requires CUDA')
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device('cuda', local_rank)
    else:
        device = choose_device(config.get('device', 'auto'))
    training_mode = config.get('training_mode', 'pretrained_finetune')
    if training_mode not in {'pretrained_finetune', 'full_downstream'}:
        raise ValueError(f'Unknown training mode: {training_mode}')
    full_training = training_mode == 'full_downstream'
    config['training_mode'] = training_mode
    if full_training:
        config.pop('pretrain_ckpt', None)
        config['finetune_1d_layers'] = config['finetune_2d_layers'] = -1
    
    config['model']['smiles']['model_name_or_path'] = _local_chemberta_path()
    seed_all(config['seed'] + rank)
    if is_main:
        print(f'ChemBERTa local weights: {config["model"]["smiles"]["model_name_or_path"]}', flush=True)
    dataset = MultiviewDataset(path_from_root(config['data_file']))
    pretrain = {} if full_training else torch.load(path_from_root(config['pretrain_ckpt']), map_location='cpu')
    model_config = copy.deepcopy(config['model'])
    if full_training:
        if 'dspgnn' not in model_config:
            raise ValueError('downstream requires model.dspgnn in the YAML config')
    else:
        model_config['dspgnn'] = copy.deepcopy(pretrain['config']['dspgnn'])
    if model_config['dspgnn']['pe_dim'] != dataset.metadata['gmpe_dim']:
        raise ValueError('DSPGNN PE dimension differs from the downstream data')
    gmpe = _downstream_gmpe(dataset, pretrain)
    model_config['num_tasks'] = dataset.metadata['num_tasks']
    model = MultiviewPredictor(model_config).to(device)
    if not full_training:
        model.encoder2d.load_state_dict(_pretrained_encoder2d_state(pretrain), strict=True)
    model.set_finetune(
        -1 if full_training else config.get('finetune_1d_layers', 2),
        -1 if full_training else config.get('finetune_2d_layers', 2),
    )
    if config.get('require_full_parameter_finetune', False):
        if full_training:
            raise ValueError('Full-parameter pretrained fine-tuning requires pretrain_ckpt')
        frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
        if frozen:
            raise ValueError(f'Full-parameter fine-tuning has frozen parameters: {frozen}')
    if config.get('gradient_checkpointing', False):
        model.encoder2d.gradient_checkpointing = True
        model.encoder1d.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
    model.max_attention_pairs = int(config.get('max_attention_pairs', 0))
    split_strategy = config.get('split_strategy', 'scaffold')
    if split_strategy not in {'scaffold', 'random'}:
        raise ValueError(f'Unknown split_strategy: {split_strategy}')
    split_path = path_from_root(config.get('split_file', '')) if config.get('split_file') else None
    if split_path and split_path.exists():
        splits = json.loads(split_path.read_text(encoding='utf-8'))
    else:
        splits = (scaffold_splits(dataset.records) if split_strategy == 'scaffold'
                  else _random_splits(len(dataset), config['seed']))
        if split_path and is_main:
            write_json(split_path, splits)
    if distributed:
        dist.barrier()
    validate_splits(splits, dataset, require_scaffold=split_strategy == 'scaffold')
    class_weights = None
    task_loss_info = {'name': 'masked_mse'}
    if dataset.metadata['task_type'] == 'cls':
        loss_name = config.setdefault('classification_loss', 'balanced_bce')
        if loss_name == 'balanced_bce':
            class_weights, task_loss_info = training_class_weights(
                dataset.records, splits['train'], dataset.metadata['num_tasks'])
            class_weights = class_weights.to(device)
            if is_main:
                print('Classification: masked balanced BCE; weights from training labels only', flush=True)
                absent = task_loss_info['tasks_without_both_classes']
                if absent:
                    print(f'Tasks without both training classes (unit weights): {absent}', flush=True)
        elif loss_name == 'bce':
            task_loss_info = {'name': 'masked_bce', 'normalization': 'mean_over_observed_entries'}
        else:
            raise ValueError(f'Unknown classification_loss: {loss_name}')
    global_batch_size = int(config['batch_size'])
    if distributed:
        if global_batch_size < world_size or global_batch_size % world_size:
            raise ValueError(
                f'Global batch_size ({global_batch_size}) must be divisible by world_size ({world_size})')
        local_batch_size = global_batch_size // world_size
    else:
        local_batch_size = global_batch_size
    subsets = [Subset(dataset, splits[name]) for name in ['train', 'val', 'test']]
    samplers = []
    loaders = []
    for subset, name in zip(subsets, ['train', 'val', 'test']):
        sampler = None
        if distributed:
            sampler = DistributedSampler(
                subset,
                num_replicas=world_size,
                rank=rank,
                shuffle=name == 'train',
                seed=config['seed'],
                drop_last=False,
            )
        samplers.append(sampler)
        loaders.append(DataLoader(
            subset,
            batch_size=local_batch_size,
            shuffle=(sampler is None and name == 'train'),
            sampler=sampler,
            num_workers=config.get('num_workers', 0),
            pin_memory=device.type == 'cuda',
            persistent_workers=bool(config.get('persistent_workers', True))
            if config.get('num_workers', 0) else False,
        ))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError('No trainable parameters')
    backbone_lr = float(config['lr'])
    head_lr = float(config.get('head_lr', backbone_lr))
    backbone_ids = {id(p) for module in (model.encoder1d.encoder, model.encoder2d)
                    for p in module.parameters()}
    parameter_groups = [
        {'params': [p for p in trainable if id(p) in backbone_ids], 'lr': backbone_lr,
         'group_name': 'pretrained_backbones'},
        {'params': [p for p in trainable if id(p) not in backbone_ids], 'lr': head_lr,
         'group_name': 'new_heads'},
    ]
    optimizer = torch.optim.AdamW(parameter_groups, lr=backbone_lr,
                                  weight_decay=config.get('weight_decay', 0))
    warmup_epochs = int(config.get('warmup_epochs', 0))
    if warmup_epochs:
        if not 0 < warmup_epochs < config['epochs']:
            raise ValueError('warmup_epochs must lie between zero and epochs')
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_epochs),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config['epochs'] - warmup_epochs)],
            milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    output = path_from_root(config['out_dir']); output.mkdir(parents=True, exist_ok=True)
    if is_main and full_training and not resume and (output / 'latest.pt').exists():
        raise FileExistsError('Run already exists; supply --resume or choose another --out_dir')
    if distributed:
        dist.barrier()
    start, best_score, best_test_score, history = 0, None, None, []
    if resume:
        state = torch.load(path_from_root(resume), map_location='cpu')
        if dataset.metadata['task_type'] == 'cls' and state.get('task_loss') != task_loss_info:
            raise ValueError(
                'Checkpoint classification loss/weights differ or are undocumented. '
                'Start a new run in a new out_dir; legacy unweighted runs cannot '
                'be silently resumed as weighted-BCE experiments.')
        if state['config']['training_mode'] != training_mode:
            raise ValueError('Resume checkpoint training mode differs from this run')
        if state['data_identity'] != dataset.metadata:
            raise ValueError('Prepared dataset metadata differs from checkpoint')
        if state['model_config'] != model_config:
            raise ValueError('Model configuration differs from checkpoint; use the original config or a new output directory')
        if state['splits'] != splits:
            raise ValueError('Resume train/validation/test splits differ from checkpoint')
        model.load_state_dict(state['model'], strict=True)
        
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); start = state['epoch']; best_score = state['best_score']
        history = state['history']; restore_rng(state['rng'])
        best_test_score = state['best_test_score']
    if is_main and config.get('require_full_parameter_finetune', False) and not resume:
        source_2d = _pretrained_encoder2d_state(pretrain)
        source_1d = torch.load(Path(config['model']['smiles']['model_name_or_path']) / 'pytorch_model.bin',
                               map_location='cpu')
        loaded_1d = model.encoder1d.encoder.state_dict()
        mismatched_1d, verified_1d, source_only_unused_pooler = [], 0, []
        for name, tensor in loaded_1d.items():
            source_name = name if name in source_1d else 'roberta.' + name
            if source_name not in source_1d and name.startswith('pooler.'):
                source_only_unused_pooler.append(name)
                continue
            if source_name not in source_1d or not torch.equal(tensor.cpu(), source_1d[source_name]):
                mismatched_1d.append(name)
            else:
                verified_1d += tensor.numel()
        exact_2d = all(torch.equal(value.cpu(), source_2d[name])
                       for name, value in model.encoder2d.state_dict().items())
        if mismatched_1d or not exact_2d:
            raise ValueError(f'Pretrained initialization audit failed: ChemBERTa={mismatched_1d}, DSPGNN={exact_2d}')
        audit = {
            'pretrain_checkpoint': str(path_from_root(config['pretrain_ckpt'])),
            'pretrain_checkpoint_sha256': file_digest(path_from_root(config['pretrain_ckpt'])),
            'chemberta_checkpoint': str(Path(config['model']['smiles']['model_name_or_path']) / 'pytorch_model.bin'),
            'chemberta_checkpoint_sha256': file_digest(Path(config['model']['smiles']['model_name_or_path']) / 'pytorch_model.bin'),
            'dspgnn_initialization_matches_checkpoint': exact_2d,
            'chemberta_verified_pretrained_parameters': verified_1d,
            'unused_pooler_not_in_mlm_checkpoint': source_only_unused_pooler,
            'frozen_parameters': [],
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in trainable),
            'encoder_parameters': {label: {'total': sum(p.numel() for p in module.parameters()),
                                         'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)}
                                   for label, module in [('chemberta', model.encoder1d), ('dspgnn', model.encoder2d)]},
            'backbone_lr': backbone_lr, 'head_lr': head_lr,
            'gradient_checkpointing': bool(config.get('gradient_checkpointing', False)),
            'max_attention_pairs': model.max_attention_pairs,
        }
        write_json(output / 'initialization_audit.json', audit)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
    if is_main:
        write_json(output / 'config.json', config)
        write_json(output / 'splits.json', splits)
        write_json(output / 'task_loss.json', task_loss_info)
        print(f'Multiview training: mode={training_mode}; device={device}; '
              f'world_size={world_size}; '
              f'batch_size={global_batch_size} (per_rank={local_batch_size}); '
              f'train/val/test={list(map(lambda x: len(splits[x]), ["train", "val", "test"]))}', flush=True)
        print(f'Parameters: total={sum(p.numel() for p in model.parameters()):,}; '
              f'trainable={sum(p.numel() for p in trainable):,}; '
              f'DSPGNN initialization={"random" if full_training else "2D/3D pretrained"}', flush=True)
    for epoch in range(start, config['epochs']):
        epoch_start = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        if distributed:
            samplers[0].set_epoch(epoch)
        train_start = time.perf_counter()
        model.train()
        sums = {'task_loss': 0., 'mvfr_loss': 0., 'loss': 0.}
        seen = 0
        total_steps = min(len(loaders[0]), config['max_steps']) if config.get('max_steps', 0) else len(loaders[0])
        progress = tqdm(total=total_steps, desc=f'Epoch {epoch + 1}/{config["epochs"]} [train]',
                        dynamic_ncols=True, disable=not is_main)
        for step, batch in enumerate(loaders[0]):
            batch = batch.to(device)
            output_batch = model(batch, regularize=True, perturbation=config['perturbation'],
                                 temperature=config['temperature'])
            task = masked_task_loss(output_batch['prediction'], batch.y, dataset.metadata['task_type'],
                                    class_weights, distributed_normalization=distributed)
            total = task + config['lambda_mvfr'] * output_batch['mvfr_loss']
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite fine-tuning loss')
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            size = batch.num_graphs
            seen += size
            for key, value in [('task_loss', task), ('mvfr_loss', output_batch['mvfr_loss']), ('loss', total)]:
                sums[key] += value.item() * size
            progress.set_postfix({key: f'{value / seen:.4f}' for key, value in sums.items()})
            progress.update(1)
            if config.get('max_steps', 0) and step + 1 >= config['max_steps']:
                break
        progress.close()
        _sync_cuda(device)
        train_seconds = time.perf_counter() - train_start
        scheduler.step()
        sums, seen = _reduce_sum_count(sums, seen, device, distributed)
        train_stats = {key: value / max(1, seen) for key, value in sums.items()}
        val_start = time.perf_counter()
        val_loss, val_pred, val_y = evaluate(
            model, loaders[1], device, dataset.metadata['task_type'], config.get('max_steps', 0),
            distributed, world_size, class_weights)
        _sync_cuda(device)
        val_seconds = time.perf_counter() - val_start
        test_start = time.perf_counter()
        test_loss, test_pred, test_y = evaluate(
            model, loaders[2], device, dataset.metadata['task_type'], config.get('max_steps', 0),
            distributed, world_size, class_weights)
        _sync_cuda(device)
        test_seconds = time.perf_counter() - test_start
        val_metric = metric(val_pred, val_y, dataset.metadata['task_type'])
        test_metric = metric(test_pred, test_y, dataset.metadata['task_type'])
        task_type = dataset.metadata['task_type']
        score = _selection_score(val_metric, task_type)
        history.append({'epoch': epoch + 1, 'train': train_stats, 'val_loss': val_loss,
                        'test_loss': test_loss, 'val': val_metric, 'test': test_metric,
                        'learning_rates': [group['lr'] for group in optimizer.param_groups],
                        'train_seconds': train_seconds, 'val_seconds': val_seconds,
                        'test_seconds': test_seconds,
                        'train_val_test_seconds': time.perf_counter() - epoch_start,
                        'peak_allocated_MiB': _peak_allocated_mib(device, distributed)})
        improved = best_score is None or score > best_score
        best_score = score if best_score is None else max(best_score, score)
        test_score = _selection_score(test_metric, task_type)
        test_improved = best_test_score is None or test_score > best_test_score
        best_test_score = test_score if best_test_score is None else max(best_test_score, test_score)
        raw_model = model.module if distributed else model
        state = {'format': 'multiview-finetune-v1', 'model': raw_model.state_dict(),
                 'encoder1d': raw_model.encoder1d.state_dict(), 'encoder2d': raw_model.encoder2d.state_dict(),
                 'fingerprint': raw_model.encoder_fp.state_dict(), 'prediction_head': raw_model.prediction_head.state_dict(),
                 'mvfr': raw_model.mvfr.state_dict(), 'model_config': model_config, 'config': config,
                 'epoch': epoch + 1, 'best_score': best_score, 'best_test_score': best_test_score,
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'splits': splits,
                 'data_identity': dataset.metadata, 'task_type': dataset.metadata['task_type'],
                 'task_loss': task_loss_info,
                 'gmpe': gmpe, 'history': history}
        if is_main:
            checkpoint_interval = max(1, int(config.get('checkpoint_interval', 1)))
            if (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == config['epochs']:
                torch.save(state, output / 'latest.pt')
            export_state = state
            if config.get('compact_best_checkpoint', False):
                export_state = {key: state[key] for key in (
                    'format', 'model', 'model_config', 'config', 'epoch', 'best_score',
                    'best_test_score', 'splits', 'data_identity', 'task_type', 'task_loss', 'gmpe')}
            if improved:
                torch.save(export_state, output / 'best.pt')
                if config.get('save_best_predictions', False):
                    _save_predictions(
                        output / 'best_predictions.pt', epoch + 1, test_pred, test_y,
                        val_metric, test_metric, splits['test'], 'validation')
            if config.get('save_best_test_checkpoint', False) and test_improved:
                torch.save(export_state, output / 'best_test.pt')
                if config.get('save_best_predictions', False):
                    _save_predictions(
                        output / 'best_test_predictions.pt', epoch + 1, test_pred, test_y,
                        val_metric, test_metric, splits['test'], 'test_search_maximum')
            write_json(output / 'history.json', history)
            print(json.dumps(history[-1]), flush=True)
        if distributed:
            dist.barrier()
    if is_main:
        print(f'Best checkpoint: {output / "best.pt"}', flush=True)
    if distributed:
        dist.destroy_process_group()
    return output / 'best.pt'


def main():
    ap = argparse.ArgumentParser(description='FP + ChemBERTa2 + DSPGNN')
    ap.add_argument('--config', default='configs/finetune_multiview.yaml')
    ap.add_argument('--data_file'); ap.add_argument('--pretrain_ckpt'); ap.add_argument('--out_dir')
    ap.add_argument('--split_file'); ap.add_argument('--split_strategy', choices=['scaffold', 'random'])
    ap.add_argument('--device'); ap.add_argument('--epochs', type=int); ap.add_argument('--batch_size', type=int)
    ap.add_argument('--num_workers', type=int); ap.add_argument('--max_steps', type=int); ap.add_argument('--resume')
    args = ap.parse_args(); config = load_config(args.config)
    config.update({key: value for key, value in vars(args).items() if key not in ['config', 'resume'] and value is not None})
    run(config, args.resume)


if __name__ == '__main__': main()
