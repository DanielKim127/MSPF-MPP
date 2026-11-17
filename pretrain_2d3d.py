import argparse
import copy
import json
import os
import sys
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from src.datasets.pcqm4mv2_multiview import PairedMoleculeDataset
from src.modules.pretrain_2d3d import Pretrain2D3D
from src.utils.multiview import (load_config, path_from_root, seed_all, choose_device,
    build_dspgnn, build_schnet, write_json, rng_state, restore_rng, finite_gradients,
    ShardAwarePairedBatchSampler, file_digest)


class DistributedBatchSampler(Sampler):
    def __init__(self, sampler, rank, world_size):
        self.sampler, self.rank, self.world_size = sampler, rank, world_size

    def __iter__(self):
        for batch in self.sampler:
            yield batch[self.rank::self.world_size]

    def __len__(self):
        return len(self.sampler)

def _model_config(config):
    return {
        'model_type': 'Pretrain2D3D',
        'hidden': config['dspgnn']['hidden'],
        'dspgnn': copy.deepcopy(config['dspgnn']),
        'schnet': copy.deepcopy(config['schnet']),
        'objective': copy.deepcopy(config['objective']),
    }


def _cpu_state_dict(module):
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}


def _model_weights_checkpoint(model, config):
    return {
        'format': 'dspgnn-2d3d-model-v2',
        'config': _model_config(config),
        'model': _cpu_state_dict(model),
    }


def run(config, resume=None):
    config = copy.deepcopy(config)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError('requires CUDA')
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device('cuda', local_rank)
    else:
        device = choose_device(config.get('device', 'auto'))
    rank = dist.get_rank() if distributed else 0
    is_main = rank == 0
    progress_stream = sys.stderr if is_main and sys.stderr.isatty() else None
    seed_all(config['seed'] + rank)
    torch.set_num_threads(config.get('cpu_threads', 2))
    root, output = path_from_root(config['data_dir']), path_from_root(config['out_dir'])
    output.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    dataset = PairedMoleculeDataset(
        root,
        cache_shards=config.get('cache_shards', 8),
        mmap_shards=config.get('mmap_shards', False),
    )
    config['dspgnn']['pe_dim'] = dataset.manifest['gmpe_dim']

    identity = file_digest(root / 'manifest.json')
    generator = torch.Generator().manual_seed(config['seed'])
    ids = torch.randperm(len(dataset), generator=generator).tolist()
    nval = min(len(dataset) - 2, max(2, round(len(dataset) * config['val_fraction'])))
    splits = {'train': ids[nval:], 'val': ids[:nval]}
    train_sampler = ShardAwarePairedBatchSampler(
        splits['train'], dataset.ends, config['batch_size'], True, config['seed'])
    val_sampler = ShardAwarePairedBatchSampler(
        splits['val'], dataset.ends, config['batch_size'])
    loader_workers = max(0, int(config.get('num_workers', 0)))
    loader_kwargs = {
        'num_workers': loader_workers,
        'pin_memory': device.type == 'cuda',
    }
    if loader_workers:
        loader_kwargs.update({
            'persistent_workers': bool(config.get('persistent_workers', True)),
            'prefetch_factor': max(1, int(config.get('prefetch_factor', 2))),
        })
    if distributed:
        local_samplers = [DistributedBatchSampler(sampler, rank, world_size)
                          for sampler in (train_sampler, val_sampler)]
    else:
        local_samplers = [train_sampler, val_sampler]
    loaders = [DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)
               for sampler in local_samplers]
    model = Pretrain2D3D(build_dspgnn(config['dspgnn']), build_schnet(config['schnet']),
                         config['dspgnn']['hidden'], **config['objective']).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    start, best, history = 0, float('inf'), []
    if resume:
        checkpoint = torch.load(path_from_root(resume), map_location='cpu')
        if checkpoint['data_identity'] != identity or checkpoint['splits'] != splits:
            raise ValueError('Resume data/splits differ from the checkpoint')
        for key in ['dspgnn', 'schnet', 'objective', 'epochs', 'batch_size', 'lr', 'seed']:
            if checkpoint['config'][key] != config[key]:
                raise ValueError(f'Resume configuration differs: {key}; use original settings')
        (model.module if distributed else model).load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start, best, history = checkpoint['epoch'], checkpoint['best_val'], checkpoint['history']
        restore_rng(checkpoint['rng'])
    gmpe = torch.load(root / 'gmpe.pt', map_location='cpu')
    if is_main:
        write_json(output / 'config.json', config)
        write_json(output / 'splits.json', splits)
    if is_main:
        print(
        f'2D/3D pretraining: device={device}; '
        f'dataset={len(dataset)} train={len(splits["train"])} val={nval}; '
        f'batch_size={config["batch_size"]} epochs={config["epochs"]} '
        f'workers={loader_workers} cache_shards={dataset.cache_shards} '
        f'gmpe_dim={config["dspgnn"]["pe_dim"]}',
        flush=True,
    )
    if is_main:
        print(
            'Objective: '
            f'global={config["objective"]["alpha_global"]} '
            f'sub={config["objective"]["alpha_sub"]} '
            f'rec={config["objective"]["alpha_rec"]} '
            f'attr={config["objective"]["alpha_attr"]} '
            f'temperature={config["objective"]["temperature"]} '
            f'mask_ratio={config["objective"]["mask_ratio"]}',
            flush=True,
        )
    for epoch in range(start, config['epochs']):
        epoch_start = time.perf_counter()
        train_sampler.epoch = epoch
        epoch_stats = {}
        phase_counts = {}
        phase_steps = {}
        for phase, loader in zip(['train', 'val'], loaders):
            training = phase == 'train'
            model.train(training)
            
            saved_rng = rng_state() if not training else None
            if not training:
                seed_all(config['seed'] + 100000)
            sums, seen = {}, 0
            max_steps = config.get('max_steps', 0)
            progress_total = len(loader)
            if max_steps:
                progress_total = min(progress_total, max_steps)
            progress = tqdm(
                loader,
                total=progress_total,
                desc=f'Epoch {epoch + 1}/{config["epochs"]} [{phase}]',
                dynamic_ncols=True,
                mininterval=0.5,
                leave=True,
                file=progress_stream or sys.stderr,
                disable=progress_stream is None,
            )
            with torch.set_grad_enabled(training):
                for step, batch in enumerate(progress):
                    batch = batch.to(device, non_blocking=device.type == 'cuda')
                    losses = model(batch)
                    if not all(torch.isfinite(value) for value in losses.values()):
                        raise FloatingPointError(f'Nonfinite {phase} loss at epoch {epoch + 1}')
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        losses['loss'].backward()
                        finite_gradients(model)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                    loss_values = {}
                    for key, value in losses.items():
                        scalar = value.item()
                        loss_values[key] = scalar
                        sums[key] = sums.get(key, 0.0) + scalar * batch.num_graphs
                    seen += batch.num_graphs
                    if progress_stream is not None:
                        progress.set_postfix(
                            {
                                'loss': f'{loss_values["loss"]:.4f}',
                                'global': f'{loss_values["global_loss"]:.4f}',
                                'sub': f'{loss_values["sub_loss"]:.4f}',
                                'rec': f'{loss_values["rec_loss"]:.4f}',
                                'attr': f'{loss_values["attr_loss"]:.4f}',
                            },
                            refresh=False,
                        )
                    if max_steps and step + 1 >= max_steps:
                        break
            progress.close()
            if saved_rng is not None:
                restore_rng(saved_rng)
            if distributed:
                keys = sorted(sums)
                packed = torch.tensor([seen] + [sums[k] for k in keys], dtype=torch.float64, device=device)
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                seen = int(packed[0].item())
                sums = {key: packed[i + 1].item() for i, key in enumerate(keys)}
            epoch_stats[phase] = {key: value / seen for key, value in sums.items()}
            phase_counts[phase] = seen
            phase_steps[phase] = step + 1
        scheduler.step()
        history.append({'epoch': epoch + 1, **epoch_stats})
        improved = epoch_stats['val']['loss'] < best
        best = min(best, epoch_stats['val']['loss'])
        
        raw_model = model.module if distributed else model
        checkpoint = {'format': 'dspgnn-2d3d-v1', 'model': raw_model.state_dict(),
                      'encoder2d': raw_model.encoder2d.state_dict(), 'encoder3d': raw_model.encoder3d.state_dict(),
                      'config': config, 'gmpe': gmpe, 'epoch': epoch + 1, 'best_val': best,
                      'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                      'rng': rng_state(), 'splits': splits, 'data_identity': identity, 'history': history}
        if is_main:
            torch.save(checkpoint, output / 'latest.pt')
            if improved:
                torch.save(_model_weights_checkpoint(raw_model, config), output / 'best.pt')
            write_json(output / 'history.json', history)
        if distributed:
            dist.barrier()
        elapsed = time.perf_counter() - epoch_start
        train_stats, val_stats = epoch_stats['train'], epoch_stats['val']
        summary = (
            f'[Epoch {epoch + 1}/{config["epochs"]}] '
            f'train_loss={train_stats["loss"]:.5f} '
            f'(global={train_stats["global_loss"]:.5f}, '
            f'sub={train_stats["sub_loss"]:.5f}, '
            f'rec={train_stats["rec_loss"]:.5f}, '
            f'attr={train_stats["attr_loss"]:.5f}) | '
            f'val_loss={val_stats["loss"]:.5f} '
            f'(global={val_stats["global_loss"]:.5f}, '
            f'sub={val_stats["sub_loss"]:.5f}, '
            f'rec={val_stats["rec_loss"]:.5f}, '
            f'attr={val_stats["attr_loss"]:.5f}) | '
            f'best_val={best:.5f} lr={optimizer.param_groups[0]["lr"]:.3e} '
            f'time={elapsed:.1f}s '
            f'samples(train/val)={phase_counts["train"]}/{phase_counts["val"]} '
            f'steps(train/val)={phase_steps["train"]}/{phase_steps["val"]} '
            f'best={"yes" if improved else "no"}'
        )
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            summary += f' peak_gpu_mem={peak_memory:.2f}GB'
            torch.cuda.reset_peak_memory_stats(device)
        if is_main:
            print(summary, flush=True)
            print(json.dumps(history[-1]), flush=True)
    if distributed:
        dist.destroy_process_group()
    return output / 'best.pt'


def main():
    ap = argparse.ArgumentParser(description='Pretrain DSPGNN/SchNet')
    ap.add_argument('--config', default='configs/pretrain_2d3d.yaml')
    for name in ['data_dir', 'out_dir', 'device']:
        ap.add_argument('--' + name)
    for name in ['epochs', 'batch_size', 'max_steps', 'num_workers']:
        ap.add_argument('--' + name, type=int)
    ap.add_argument('--resume')
    args = ap.parse_args()
    config = load_config(args.config)
    config.update({key: value for key, value in vars(args).items()
                   if key not in ['config', 'resume'] and value is not None})
    print(f'Best checkpoint: {run(config, args.resume)}')


if __name__ == '__main__':
    main()