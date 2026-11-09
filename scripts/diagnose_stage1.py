#!/usr/bin/env python3
"""Bounded, frozen-input diagnostics of existing stage1 checkpoints; no training.

prepare tokenizes on the host; evaluate requires CUDA. Only summary is dependency-free.
Context corruption measures sensitivity, not evidence correctness or task quality.
"""
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys


ARMS = ('P0', 'B0', 'L3', 'G2')
CONDITIONS = ('full', 'far_shuffled', 'far_masked')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def bucket(length):
    return 'le512' if length <= 512 else '513to2048' if length <= 2048 else '2049to8192'


def implementation_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve()] + [root / 'src/nextbert' / name for name in
                                        ('config.py', 'model.py', 'attention.py', 'train.py', 'data.py')]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def configs(stage):
    from nextbert.config import Config
    result = {arm: Config.load(stage / f'{arm}.json') for arm in ARMS if arm != 'P0'}
    for arm, config in result.items():
        if config.variant != arm or config.task != 'mlm' or len(config.revision) != 40:
            raise ValueError('Require pinned stage1 MLM configs with matching variants')
        expected = replace(result['B0'], variant=arm, output_dir=config.output_dir)
        if config.to_dict() != expected.to_dict():
            raise ValueError('Stage arms differ in more than variant/output directory')
    return result


def prepare(a):
    from transformers import AutoTokenizer
    stage, output = Path(a.stage), Path(a.output)
    if output.exists():
        raise FileExistsError('Use a fresh diagnostic directory; existing inputs are never overwritten')
    cfg = configs(stage)['B0']
    frozen = read(stage / 'stage-manifest.json')
    if frozen['model_revision'] != cfg.revision:
        raise ValueError('Stage manifest/config revision mismatch')
    tok = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.revision, use_fast=True)
    if tok.mask_token_id is None:
        raise ValueError('Tokenizer has no mask token')
    special_ids = set(tok.all_special_ids)
    sources, selected, long_selected, source_counts = {}, [], [], {}
    for lang in ('zh', 'en'):
        path = stage / f'dev.{lang}.jsonl'
        sources[str(path)] = digest(path)
        if sources[str(path)] != frozen['dev_file_sha256'][lang]:
            raise ValueError('Stage dev data changed')
        groups, long_docs, seen = defaultdict(list), [], set()
        for ex in rows(path):
            if ex['id'] in seen or ex['language'] != lang:
                raise ValueError('Duplicate document ID or incorrect language')
            seen.add(ex['id'])
            encoded = tok(ex['text'], add_special_tokens=True, truncation=True, max_length=8192,
                          return_special_tokens_mask=True, return_attention_mask=False,
                          return_token_type_ids=False)
            doc = {'id': ex['id'], 'language': lang, 'text': ex['text'], 'encoded': encoded,
                   'natural_length': len(encoded['input_ids'])}
            groups[bucket(doc['natural_length'])].append(doc)
            if doc['natural_length'] >= a.min_long:
                long_docs.append(doc)
        source_counts[lang] = {key: len(value) for key, value in groups.items()}
        for key in ('le512', '513to2048', '2049to8192'):
            values = sorted(groups[key], key=lambda x: x['id'])
            random.Random(f'{a.seed}:{lang}:{key}').shuffle(values)
            selected.extend(values[:a.per_bucket])
        long_docs.sort(key=lambda x: x['id'])
        random.Random(f'{a.seed}:{lang}:long').shuffle(long_docs)
        if len(long_docs) < a.long_per_language:
            raise ValueError(f'{lang}: need {a.long_per_language} documents >= {a.min_long} tokens; '
                             f'only {len(long_docs)} available. Lower --long-per-language explicitly.')
        long_selected.extend(long_docs[:a.long_per_language])

    def pack(doc, limit):
        # Use the fast tokenizer's postprocessor, including its own special tokens;
        # generic prepare_for_model/build_inputs_with_special_tokens is not portable
        # across fast tokenizer classes.
        packed = doc['encoded'] if limit == 8192 else tok(
            doc['text'], add_special_tokens=True, truncation=True, max_length=limit,
            return_special_tokens_mask=True, return_attention_mask=False, return_token_type_ids=False)
        available = [i for i, flag in enumerate(packed['special_tokens_mask'])
                     if not flag and packed['input_ids'][i] not in special_ids]
        if not available:
            raise ValueError('No maskable content tokens')
        return packed['input_ids'], available

    cases = []

    def add(doc, ids, available, targets, probe, condition, group, **meta):
        labels = [-100] * len(ids)
        for i, gold in targets:
            labels[i] = gold
        cases.append({'id': f'{group}/{condition}', 'base_id': doc['id'], 'group_id': group,
                      'language': doc['language'], 'probe': probe, 'condition': condition,
                      'natural_length': doc['natural_length'], 'natural_bucket': bucket(doc['natural_length']),
                      'actual_length': len(ids), 'input_ids': ids, 'content_positions': available,
                      'labels': labels, 'target_signature': signature(targets), **meta})

    for doc in selected:
        used_lengths = set()
        for limit in (512, 2048, 8192):
            ids, available = pack(doc, limit)
            if len(ids) in used_lengths:
                continue
            used_lengths.add(len(ids))
            rng = random.Random(f'{a.seed}:length:{doc["id"]}:{limit}')
            positions = sorted(rng.sample(available, max(1, round(len(available) * cfg.mask_probability))))
            targets = [(i, ids[i]) for i in positions]
            corrupted = ids.copy()
            for i in positions:
                op = rng.random()
                if op < .8:
                    corrupted[i] = tok.mask_token_id
                elif op < .9:
                    replacement = rng.randrange(len(tok))
                    while replacement in special_ids:
                        replacement = rng.randrange(len(tok))
                    corrupted[i] = replacement
            add(doc, corrupted, available, targets, 'length', 'full', f'{doc["id"]}/length/{limit}',
                context_limit=limit, position='mixed')
    for doc in long_selected:
        ids, available = pack(doc, 8192)
        for position, fraction in (('start', .1), ('middle', .5), ('end', .9)):
            center = available[min(len(available) - 1, int(len(available) * fraction))]
            pool = [i for i in available if abs(i - center) <= 32]
            rng = random.Random(f'{a.seed}:context:{doc["id"]}:{position}')
            chosen = sorted(rng.sample(pool, min(16, len(pool))))
            targets = [(i, ids[i]) for i in chosen]
            masked = ids.copy()
            for i in chosen:
                masked[i] = tok.mask_token_id
            # Keep target positions, all tokens inside the local neighborhood,
            # sequence length, special tokens, and attention/content masks fixed.
            far = [i for i in available if i < min(chosen) - a.local_radius
                   or i > max(chosen) + a.local_radius]
            if not far:
                raise ValueError('Context probe has no far tokens')
            shuffled = masked.copy()
            values = [ids[i] for i in far]
            rng.shuffle(values)
            for i, value in zip(far, values):
                shuffled[i] = value
            removed = masked.copy()
            for i in far:
                removed[i] = tok.mask_token_id
            for condition, corrupted in zip(CONDITIONS, (masked, shuffled, removed)):
                add(doc, corrupted, available, targets, 'context', condition,
                    f'{doc["id"]}/context/{position}', context_limit=8192, position=position,
                    target_positions=chosen, far_tokens=len(far))

    if not cases or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Empty or duplicate diagnostic cases')
    output.mkdir(parents=True)
    case_path = output / 'cases.jsonl'
    case_path.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
    cfg_hashes = {arm: digest(stage / f'{arm}.json') for arm in ARMS if arm != 'P0'}
    manifest = {'version': 1, 'seed': a.seed, 'stage': str(stage), 'model_name': cfg.model_name,
                'implementation_sha256': implementation_hashes(),
                'model_revision': cfg.revision, 'source_sha256': sources,
                'stage_config_sha256': cfg_hashes, 'cases_sha256': digest(case_path),
                'per_bucket_cap': a.per_bucket, 'long_per_language': a.long_per_language,
                'min_long': a.min_long, 'local_radius': a.local_radius,
                'source_bucket_counts': source_counts, 'case_count': len(cases),
                'length_document_count': len(selected), 'context_document_count': len(long_selected),
                'max_forward_calls_all_arms': len(cases) * len(ARMS),
                'notes': ['P0 is the unadapted pretrained B0 through the same encoder implementation.',
                          'Length probe uses 15% targets and 80/10/10 corruption; prefixes have different targets.',
                          'Context probe masks only fixed targets, then shuffles or masks far content tokens.',
                          'far_masked does NOT delete positions or hide them from attention; both corruptions can be OOD.',
                          'Context sensitivity is not proof of useful evidence, retention, or downstream quality.',
                          'Small deterministic natural-length strata; not a new representative benchmark.']}
    write(output / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def evaluate(a):
    import torch
    from torch.nn import functional as F
    from nextbert.data import to_device
    from nextbert.train import autocast, build, load_weights
    output = Path(a.output)
    manifest_path = output / 'manifest.json'
    manifest = read(manifest_path)
    if implementation_hashes() != manifest['implementation_sha256']:
        raise ValueError('Diagnostic/model implementation changed; prepare a fresh output directory')
    if digest(output / 'cases.jsonl') != manifest['cases_sha256']:
        raise ValueError('Frozen diagnostic cases changed')
    stage = Path(manifest['stage'])
    for arm, value in manifest['stage_config_sha256'].items():
        if digest(stage / f'{arm}.json') != value:
            raise ValueError('Stage config changed after freezing')
    cfgs = configs(stage)
    checkpoint = None if a.arm == 'P0' else Path(cfgs[a.arm].output_dir) / 'final'
    identity = {'arm': a.arm, 'manifest_sha256': digest(manifest_path),
                'checkpoint_sha256': digest(checkpoint / 'checkpoint.pt') if checkpoint else None}
    result_path = output / f'{a.arm}-metrics.json'
    predictions = output / f'{a.arm}-predictions.jsonl'
    if result_path.exists():
        result = read(result_path)
        if result['identity'] != identity or digest(predictions) != result['predictions_sha256']:
            raise ValueError('Existing result identity changed; use a fresh output directory')
        print(f'{a.arm}: complete result already exists; preserving it', flush=True)
        return
    if a.arm == 'P0':
        config = replace(cfgs['B0'], lora_rank=0, checkpointing=False)
        model, tok = build(config)
    else:
        model, tok, config, payload = load_weights(checkpoint)
        del payload
        if config.to_dict() != cfgs[a.arm].to_dict():
            raise ValueError('Checkpoint config differs from frozen stage configuration')
    if (config.model_name, config.revision) != (manifest['model_name'], manifest['model_revision']):
        raise ValueError('Model identity differs from the frozen input tokenizer')
    del tok
    model.eval()
    gate_stats, handles = {}, []
    current = ['']

    def gate_hook(name):
        def collect(_module, _inputs, logits):
            values = logits.float().sigmoid()
            key = f'{name}/{current[0]}'
            vector = torch.stack((values.sum(), (values * values).sum(),
                                  (values < .01).float().sum(), (values > .1).float().sum()))
            if key not in gate_stats:
                gate_stats[key] = [vector, values.numel()]
            else:
                gate_stats[key][0] += vector
                gate_stats[key][1] += values.numel()
        return collect

    for name, branch in model.branches.items():
        if hasattr(branch, 'gate'):
            handles.append(branch.gate.register_forward_hook(gate_hook(name)))
    temp = predictions.with_suffix('.jsonl.tmp')
    count = 0
    with torch.inference_mode(), temp.open('w') as handle:
        for case in rows(output / 'cases.jsonl'):
            content = torch.zeros((1, case['actual_length']), dtype=torch.bool)
            content[0, case['content_positions']] = True
            batch = to_device({'input_ids': torch.tensor([case['input_ids']], dtype=torch.long),
                               'labels': torch.tensor([case['labels']], dtype=torch.long),
                               'content_mask': content, 'attention_mask': torch.ones_like(content, dtype=torch.long)}, 'cuda')
            current[0] = f'{case["probe"]}/{case["context_limit"]}/{case["condition"]}'
            with autocast(config):
                _, logits = model(batch)
            gold = batch['labels'][batch['labels'] != -100]
            nll = F.cross_entropy(logits.float(), gold, reduction='none')
            if not bool(torch.isfinite(nll).all()):
                raise FloatingPointError('Non-finite diagnostic loss')
            record = {k: v for k, v in case.items() if k not in ('input_ids', 'labels', 'content_positions')}
            record.update(nll_sum=float(nll.sum()), masked_tokens=gold.numel(),
                          correct_tokens=int((logits.argmax(-1) == gold).sum()), loss=float(nll.mean()))
            handle.write(json.dumps(record) + '\n')
            count += 1
            if count % 25 == 0:
                print(f'{a.arm}: {count}/{manifest["case_count"]} forward passes', flush=True)
    for handle in handles:
        handle.remove()
    if count != manifest['case_count']:
        raise ValueError('Incomplete diagnostic case file')
    temp.replace(predictions)
    gates = {}
    for key, (vector, n) in gate_stats.items():
        total, squares, low, high = vector.cpu().tolist()
        gates[key] = {'observations': n, 'mean': total / n,
                      'std': math.sqrt(max(0., squares / n - (total / n) ** 2)),
                      'fraction_lt_0.01': low / n, 'fraction_gt_0.1': high / n}
    write(result_path, {'identity': identity, 'records': count, 'predictions_sha256': digest(predictions),
                        'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(), 'gate_activations': gates,
                        'note': 'No optimizer, parameter update, throughput benchmark, or downstream task evaluation.'})
    print(f'{a.arm}: complete', flush=True)


def summarize(a):
    output = Path(a.output)
    manifest = read(output / 'manifest.json')
    predictions, aggregates = {}, {}
    for arm in ARMS:
        metrics = read(output / f'{arm}-metrics.json')
        path = output / f'{arm}-predictions.jsonl'
        if (metrics['identity']['arm'] != arm
                or metrics['identity']['manifest_sha256'] != digest(output / 'manifest.json')
                or digest(path) != metrics['predictions_sha256']):
            raise ValueError('Result integrity check failed')
        rr = list(rows(path))
        if len(rr) != manifest['case_count'] or len({r['id'] for r in rr}) != len(rr):
            raise ValueError('Incomplete/duplicate prediction records')
        predictions[arm] = {r['id']: r for r in rr}
        groups = defaultdict(list)
        for r in rr:
            key = (r['probe'], r['language'], r['context_limit'], r['position'], r['condition'])
            groups[key].append(r)
        aggregates[arm] = []
        for key, values in sorted(groups.items()):
            n = sum(r['masked_tokens'] for r in values)
            aggregates[arm].append(dict(zip(('probe', 'language', 'context_limit', 'position', 'condition'), key),
                                        documents=len(values), masked_tokens=n,
                                        mean_actual_length=sum(r['actual_length'] for r in values) / len(values),
                                        document_macro_loss=sum(r['loss'] for r in values) / len(values),
                                        token_weighted_loss=sum(r['nll_sum'] for r in values) / n,
                                        masked_token_accuracy=sum(r['correct_tokens'] for r in values) / n))
    base = predictions['P0']
    for arm in ARMS:
        if predictions[arm].keys() != base.keys():
            raise ValueError('Arm case IDs differ')
        for key, r in predictions[arm].items():
            if (r['target_signature'], r['masked_tokens'], r['actual_length']) != (
                    base[key]['target_signature'], base[key]['masked_tokens'], base[key]['actual_length']):
                raise ValueError('Unpaired targets across arms')

    def paired_interval(pairs):
        # Cluster by source document: positions and repeated prefixes are dependent.
        groups = defaultdict(list)
        for doc_id, delta in pairs:
            groups[doc_id].append(delta)
        values = list(groups.values())
        rng = random.Random(manifest['seed'])
        boots = []
        for _ in range(5000):
            chosen = [values[rng.randrange(len(values))] for _ in values]
            boots.append(sum(map(sum, chosen)) / sum(map(len, chosen)))
        boots.sort()
        return {'mean_delta': sum(d for _, d in pairs) / len(pairs), 'groups': len(values),
                'paired_cases': len(pairs), 'cluster_bootstrap_ci95': [boots[124], boots[4874]]}

    differences = []
    for left, right in (('B0', 'P0'), ('G2', 'B0'), ('L3', 'B0'), ('G2', 'L3')):
        groups = defaultdict(list)
        for key, r in predictions[left].items():
            group = (r['probe'], r['language'], r['context_limit'], r['condition'])
            groups[group].append((r['base_id'], r['loss'] - predictions[right][key]['loss']))
        for key, pairs in sorted(groups.items()):
            differences.append({'comparison': f'{left}-{right}',
                                **dict(zip(('probe', 'language', 'context_limit', 'condition'), key)),
                                **paired_interval(pairs)})
    sensitivity = []
    for arm in ARMS:
        groups = defaultdict(list)
        for r in predictions[arm].values():
            if r['probe'] != 'context' or r['condition'] == 'full':
                continue
            full = predictions[arm][f'{r["group_id"]}/full']
            if r['target_signature'] != full['target_signature']:
                raise ValueError('Context intervention changed target labels/positions')
            groups[(r['language'], r['condition'])].append((r['base_id'], r['loss'] - full['loss']))
        for (lang, condition), pairs in groups.items():
            sensitivity.append({'arm': arm, 'language': lang, 'condition': condition,
                                'definition': 'corrupted far context loss minus full context loss',
                                **paired_interval(pairs)})
    write(output / 'summary.json', {'manifest': manifest, 'aggregates': aggregates,
                                  'model_differences': differences, 'context_sensitivity': sensitivity,
                                  'limitations': ['One training seed and one frozen diagnostic corruption seed.',
                                                  'Bootstrap clusters source documents; no multiple-comparison correction.',
                                                  'Positive context delta indicates sensitivity, not correct long-range reasoning.',
                                                  'P0 comparisons only probe MLM retention in this corpus, not original task capabilities.',
                                                  'No timing claims; gate instrumentation adds overhead.']})
    print(f'Saved {output / "summary.json"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', required=True, choices=('prepare', 'evaluate', 'summary', 'all'))
    p.add_argument('--stage', default='outputs/stage1-s17')
    p.add_argument('--output', default='outputs/stage1-diagnostics-s1701')
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--per-bucket', type=int, default=12)
    p.add_argument('--long-per-language', type=int, default=8)
    p.add_argument('--min-long', type=int, default=4096)
    p.add_argument('--local-radius', type=int, default=256)
    a = p.parse_args()
    if min(a.per_bucket, a.long_per_language, a.local_radius) < 1 or not 1024 <= a.min_long <= 8192:
        p.error('Invalid document cap, radius, or minimum long length')
    if a.action == 'prepare':
        prepare(a)
    elif a.action == 'evaluate':
        if a.arm is None:
            p.error('evaluate requires --arm')
        evaluate(a)
    elif a.action == 'summary':
        summarize(a)
    else:
        # Frozen manifest is authoritative when resuming; never silently re-prepare.
        manifest = read(Path(a.output) / 'manifest.json')
        print(f'Frozen plan: {manifest["case_count"]} cases x {len(ARMS)} arms; no training', flush=True)
        for arm in ARMS:
            subprocess.run([sys.executable, __file__, '--action', 'evaluate', '--arm', arm,
                            '--output', a.output], check=True)
        summarize(a)


if __name__ == '__main__':
    main()
