import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import torch
from src.datasets.moleculenet import DATASET_SPECS, _infer_label_columns
from src.utils.multiview import GMPEGenerator, gmpe_payload, path_from_root, tensor_record, write_json
from src.utils.conformers import molecule_graph
from src.utils.fingerprints import fingerprint, FP_SCHEMA


def prepare(dataset, csv_path, output, gmpe_ckpt, limit=0, selected_ids=None, label_columns=None):
    csv_path, output = path_from_root(csv_path), path_from_root(output)
    if output.exists():
        raise FileExistsError(f'Prepared file already exists: {output}; choose a new --out')
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    spec = DATASET_SPECS.get(dataset, {'task_type': 'cls' if dataset == 'MUV' else None, 'label_cols': None})
    if not spec['task_type']:
        raise ValueError(f'Unknown dataset: {dataset}')
    columns = label_columns or _infer_label_columns(df, dataset, spec)
    labels = df[columns].apply(pd.to_numeric, errors='coerce')
    if spec['task_type'] == 'cls':
        labels = labels.where(labels >= 0)
        valid_values = labels.stack().unique()

    generator = GMPEGenerator(gmpe_payload(gmpe_ckpt))
    ids = list(selected_ids) if selected_ids is not None else list(range(min(len(df), limit or len(df))))
    records, failures = [], []
    for gid in ids:
        try:
            graph, _ = molecule_graph(str(df.iloc[gid].smiles), labels.iloc[gid].to_numpy(dtype=float), gid)
            graph.graph_id = gid
            graph.pe = generator(graph)
            graph.fp = fingerprint(graph.smiles)
            records.append(tensor_record(graph))
        except Exception as exc:
            failures.append({'gid': gid, 'error': str(exc)})
        if len(records) and len(records) % 100 == 0:
            print(f'Prepared {len(records)} records', flush=True)
    if not records:
        raise ValueError(f'No valid samples; failures: {failures[:3]}')
    metadata = {'dataset': dataset, 'task_type': spec['task_type'], 'num_tasks': len(columns),
                'label_columns': columns, 'fp_schema': FP_SCHEMA, 'gmpe_dim': generator.payload['config']['hidden'],
                'source_csv': str(csv_path), 'retained': len(records), 'failed': len(failures)}
    torch.save({'records': records, 'metadata': metadata, 'gmpe': generator.payload}, output)
    write_json(output.with_suffix('.failures.json'), failures)
    print(f'Prepared {len(records)}/{len(ids)} downstream samples: {output}', flush=True)
    return output


def main():
    ap = argparse.ArgumentParser(description='Prepare FP/SMILES/2D/GMPE')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--gmpe_ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--label_columns', nargs='+')
    args = ap.parse_args()
    prepare(args.dataset, args.csv, args.out, args.gmpe_ckpt, args.limit, label_columns=args.label_columns)


if __name__ == '__main__':
    main()