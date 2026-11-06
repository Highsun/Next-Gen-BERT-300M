#!/usr/bin/env python3
"""Paired, cluster bootstrap over per-example/per-query JSONL; no seed-level claims."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', required=True)
parser.add_argument('--candidate', required=True)
parser.add_argument('--metric', default='correct')
parser.add_argument('--cluster-key', default='base_id')
parser.add_argument('--samples', type=int, default=10000)
parser.add_argument('--seed', type=int, default=17)
parser.add_argument('--output', required=True)
args = parser.parse_args()

def read(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row['id'] in result:
            raise ValueError('Duplicate ids')
        result[row['id']] = row
    return result

base, candidate = read(args.baseline), read(args.candidate)
if not base or base.keys() != candidate.keys():
    raise ValueError('Require identical nonempty example IDs')
groups = defaultdict(list)
for key in base:
    groups[str(base[key].get(args.cluster_key, key))].append(float(candidate[key][args.metric])-float(base[key][args.metric]))
values = [np.asarray(v) for v in groups.values()]
rng = np.random.default_rng(args.seed)
deltas = []
for _ in range(args.samples):
    indices = rng.integers(len(values), size=len(values))
    deltas.append(float(np.concatenate([values[i] for i in indices]).mean()))
Path(args.output).write_text(json.dumps({'mean_delta': float(np.concatenate(values).mean()),
    'paired_cluster_interval_95': np.quantile(deltas, [.025, .975]).tolist(), 'clusters': len(values),
    'note': 'Sampling CI only; report training seeds separately. Holm correction not performed here.'}, indent=2))
