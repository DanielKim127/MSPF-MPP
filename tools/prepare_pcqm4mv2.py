import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import torch
from src.utils.multiview import GMPEGenerator, gmpe_payload, path_from_root, tensor_record, write_json
from src.utils.conformers import molecule_graph, generate_conformer


def _prepare_record(task):
    gid, smiles, seed, pe_mode, pe_dim, gmpe_cache = task
    try:
        graph, mol = molecule_graph(str(smiles), gid=gid)
        graph.pos = generate_conformer(mol, seed + gid)
        if pe_mode == 'gmpe':
            embedding_path = Path(gmpe_cache) / f'{gid}.pt'
            graph.pe = torch.load(embedding_path, map_location='cpu')['pe'].float()
            if graph.pe.size(0) != graph.num_nodes or graph.pe.size(1) != pe_dim:
                raise ValueError(f'GMPE cache shape mismatch for gid={gid}')
        elif pe_mode == 'rwpe':
            from src.transforms.RWPE import RandomWalkPE
            graph.pe = RandomWalkPE(K=pe_dim, normalise=True)(graph).pe
        else:
            graph.pe = torch.zeros((graph.num_nodes, pe_dim), dtype=torch.float32)
        graph.graph_id = gid
        graph.atom_mapping = torch.arange(graph.num_nodes)
        return gid, tensor_record(graph), None
    except Exception as exc:
        return gid, None, str(exc)


def _init_worker():
    torch.set_num_threads(1)


def prepare(csv_path, output, gmpe_ckpt=None, limit=0, seed=42, shard_size=2048,
            dataset_name='PCQM4Mv2', selected_ids=None, pe_mode='gmpe', pe_dim=128,
            gmpe_cache=None, workers=1):
    csv_path, output = path_from_root(csv_path), path_from_root(output)
    if (output / 'manifest.json').exists():
        raise FileExistsError(f'new out_dir required')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob('shard-*.pt')):
        raise FileExistsError('use a new directory')
    pe_mode = str(pe_mode).lower()
    if pe_mode == 'gmpe':
        payload = gmpe_payload(gmpe_ckpt)
        generator = None if gmpe_cache else GMPEGenerator(payload)
        pe_dim = int(payload['config']['hidden'])
        pe_transform = None
        torch.save(payload, output / 'gmpe.pt')
    elif pe_mode == 'rwpe':
        generator = None
        pe_dim = int(pe_dim)
        from src.transforms.RWPE import RandomWalkPE
        pe_transform = RandomWalkPE(K=pe_dim, normalise=True)
        torch.save({'mode': 'rwpe', 'dim': pe_dim}, output / 'gmpe.pt')
    else:
        generator = None
        pe_dim = int(pe_dim)
        pe_transform = None
        torch.save({'mode': 'zeros', 'dim': pe_dim}, output / 'gmpe.pt')
    records, shards, attempted, kept, offset = [], [], 0, 0, 0
    failures = output / 'failures.jsonl'
    worker_count = max(1, int(workers))

    def flush_records():
        nonlocal records
        if len(records) >= shard_size:
            name = f'shard-{len(shards):05d}.pt'
            torch.save(records, output / name)
            shards.append({'file': name, 'count': len(records)})
            records = []

    with failures.open('w', encoding='utf-8') as failure_stream:
        executor = (ProcessPoolExecutor(max_workers=worker_count, initializer=_init_worker)
                    if worker_count > 1 else None)
        try:
            for frame in pd.read_csv(csv_path, chunksize=4096):
                if 'smiles' not in frame:
                    raise ValueError('CSV requires a smiles column')
                tasks = []
                for local, row in enumerate(frame.itertuples(index=False)):
                    gid = offset + local
                    if selected_ids is not None and gid not in selected_ids:
                        continue
                    if limit and attempted + len(tasks) >= limit:
                        break
                    tasks.append((gid, row.smiles, seed, pe_mode, pe_dim,
                                  str(path_from_root(gmpe_cache)) if gmpe_cache else None))
                if executor is None:
                    results = map(_prepare_record, tasks) if gmpe_cache else None
                else:
                    results = executor.map(_prepare_record, tasks, chunksize=16)
                if results is not None:
                    for gid, record, error in results:
                        attempted += 1
                        if error is not None:
                            failure_stream.write(json.dumps({'gid': gid, 'error': error}, ensure_ascii=False) + '\n')
                        else:
                            records.append(record)
                            kept += 1
                        flush_records()
                        if attempted % 100 == 0:
                            print(f'Attempted {attempted}, retained {kept}', flush=True)
                else:
                    for local, row in enumerate(frame.itertuples(index=False)):
                        gid = offset + local
                        if selected_ids is not None and gid not in selected_ids:
                            continue
                        if limit and attempted >= limit:
                            break
                        attempted += 1
                        try:
                            graph, mol = molecule_graph(str(row.smiles), gid=gid)
                            graph.pos = generate_conformer(mol, seed + gid)
                            if generator is not None:
                                graph.pe = generator(graph)
                            elif pe_transform is not None:
                                graph.pe = pe_transform(graph).pe
                            else:
                                graph.pe = torch.zeros((graph.num_nodes, pe_dim), dtype=torch.float32)
                            graph.graph_id = gid
                            graph.atom_mapping = torch.arange(graph.num_nodes)
                            records.append(tensor_record(graph))
                            kept += 1
                        except Exception as exc:
                            failure_stream.write(json.dumps({'gid': gid, 'error': str(exc)}, ensure_ascii=False) + '\n')
                        flush_records()
                        if attempted % 100 == 0:
                            print(f'Attempted {attempted}, retained {kept}', flush=True)
                offset += len(frame)
                if limit and attempted >= limit:
                    break
        finally:
            if executor is not None:
                executor.shutdown()
        if records:
            name = f'shard-{len(shards):05d}.pt'
            torch.save(records, output / name)
            shards.append({'file': name, 'count': len(records)})
            records = []
    manifest = {'schema': 'paired-heavy-atoms-v1', 'dataset': dataset_name, 'source_csv': str(csv_path),
                'attempted': attempted, 'retained': kept, 'failed': attempted - kept,
                'seed': seed, 'gmpe_dim': pe_dim, 'pe_mode': pe_mode,
                'conformer': 'ETKDGv3 + MMFF/UFF; heavy atom coordinates',
                'shards': shards}
    write_json(output / 'manifest.json', manifest)
    print(f'Prepared {kept}/{attempted} molecules: {output}', flush=True)
    return output


def main():
    ap = argparse.ArgumentParser(description='Prepare sharded paired')
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--csv')
    source.add_argument('--pcqm_root')
    ap.add_argument('--gmpe_ckpt')
    ap.add_argument('--gmpe_cache')
    ap.add_argument('--pe_mode', choices=['gmpe', 'rwpe', 'zeros'], default='gmpe')
    ap.add_argument('--pe_dim', type=int, default=128)
    ap.add_argument('--out_dir', default='data/pcqm4mv2/paired')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--shard_size', type=int, default=2048)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--dataset_name', default='PCQM4Mv2')
    args = ap.parse_args()
    if args.pcqm_root:
        
        pcqm_root = path_from_root(args.pcqm_root)
        candidates = [
            pcqm_root / 'raw' / 'pcqm4m.csv',
            pcqm_root / 'raw' / 'pcqm4m.csv.gz',
            pcqm_root / 'pcqm4m-v2' / 'raw' / 'data.csv.gz',
            pcqm_root / 'pcqm4m-v2' / 'raw' / 'data.csv',
        ]
        existing = next((candidate for candidate in candidates if candidate.exists()), None)
        if existing is None:
            from ogb.lsc import PCQM4Mv2Dataset
            dataset = PCQM4Mv2Dataset(root=str(pcqm_root), only_smiles=True)
            existing = Path(dataset.folder) / 'raw/data.csv.gz'
        args.csv = str(existing)
    prepare(args.csv, args.out_dir, args.gmpe_ckpt, args.limit, args.seed,
            args.shard_size, args.dataset_name, pe_mode=args.pe_mode, pe_dim=args.pe_dim,
            gmpe_cache=args.gmpe_cache, workers=args.workers)


if __name__ == '__main__':
    main()