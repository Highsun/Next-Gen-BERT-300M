#!/usr/bin/env python3
"""AutoDL-only real-long-document training and operator profiling.

Uses existing frozen corpus/cache. No downloads, no new quality claims, no local authoring execution.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def prepare(source, output, per_language):
    from transformers import AutoTokenizer
    from nextbert.data import rows, file_hash
    manifest = json.loads((source/'manifest.json').read_text())
    if file_hash(source/'train.jsonl') != manifest['files']['train']:
        raise ValueError('Source train SHA differs from frozen manifest')
    if output.exists():
        recorded = json.loads((output/'manifest.json').read_text())
        if (recorded['source_sha256'] != manifest['files']['train']
                or recorded['per_language'] != per_language
                or file_hash(output/'train.jsonl') != recorded['data_sha256']):
            raise ValueError('Existing long-check data differ; choose a fresh output directory')
        return recorded
    tok = AutoTokenizer.from_pretrained(manifest['model_name'], revision=manifest['model_revision'],
                                       local_files_only=True, use_fast=True)
    selected = {'zh': [], 'en': []}
    for row in rows(source/'train.jsonl'):
        language = row.get('language')
        if language not in selected or len(selected[language]) >= per_language:
            continue
        length = len(tok(row['text'], truncation=True, max_length=8192)['input_ids'])
        if length >= 8000:
            selected[language].append((row, length))
        if all(len(v) == per_language for v in selected.values()):
            break
    if any(len(v) < per_language for v in selected.values()):
        raise ValueError('Not enough >=8000-token documents; do not substitute short documents silently')
    output.mkdir(parents=True)
    examples = [selected[language][i] for i in range(per_language) for language in ('zh', 'en')]
    (output/'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r, _ in examples))
    recorded = {'source_sha256': manifest['files']['train'], 'model_revision': manifest['model_revision'],
                'per_language': per_language, 'data_sha256': file_hash(output/'train.jsonl'),
                'documents': [{'id': row['id'], 'tokens': n, 'language': row['language']} for row, n in examples],
                'purpose': 'Real 8K-adjacent memory/training check only; repeated documents, not quality evidence.'}
    for variant in ('B0', 'G2'):
        c = json.loads((source/f'{variant}-starter.json').read_text())
        if c['revision'] != manifest['model_revision']:
            raise ValueError('Base revision mismatch')
        c.update(max_length=8192, max_steps=20, accumulation=2, length_curriculum=False,
                 save_every=10, log_every=1, keep_checkpoints=2,
                 profile_pipeline=True,
                 train_file=str(output/'train.jsonl'), output_dir=f'runs/{variant}-{output.name}')
        write_json(output/f'{variant}.json', c)
    write_json(output/'manifest.json', recorded)
    return recorded


def summarize(output):
    summary = {'purpose': 'Complete optimizer-step timing, not a downstream quality comparison.'}
    for variant in ('B0', 'G2'):
        config = json.loads((output/f'{variant}.json').read_text())
        run = Path(config['output_dir'])
        records = [json.loads(line) for line in (run/'train.jsonl').read_text().splitlines() if line.strip()]
        if [r['step'] for r in records] != list(range(1, 21)) or not all(math.isfinite(r['loss']) for r in records):
            raise ValueError('Require exactly 20 finite completed steps per run')
        times = [records[i]['elapsed_seconds']-records[i-1]['elapsed_seconds'] for i in range(5, len(records))]
        summary[variant] = {'steps': len(records), 'tokens': records[-1]['tokens'],
                            'elapsed_seconds': records[-1]['elapsed_seconds'],
                            'median_step_seconds_after_step5': statistics.median(times),
                            'peak_allocated_GiB': records[-1]['max_memory_bytes']/2**30,
                            'final_loss': records[-1]['loss'],
                            'note': 'Elapsed may include intermediate checkpoint I/O; final save excluded.'}
        if all('timing' in r for r in records[5:]):
            summary[variant]['mean_phase_seconds_after_step5'] = {
                key: statistics.mean(r['timing'][key] for r in records[5:]) for key in records[5]['timing']}
            summary[variant]['timing_note'] = 'Phase boundaries synchronize CUDA; diagnostic overhead applies.'
    write_json(output/'training-summary.json', summary)
    print(json.dumps(summary, indent=2))


def profile_run(output, variant, trace):
    import torch
    from nextbert.data import Collator, rows, to_device
    from nextbert.train import load_weights, autocast, loss_for_batch
    config = json.loads((output/f'{variant}.json').read_text())
    model, tok, c, payload = load_weights(Path(config['output_dir'])/'final')
    del payload
    model.train()
    batch = to_device(Collator(tok, c)([next(rows(output/'train.jsonl'))]), 'cuda')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c.learning_rate)
    def iteration():
        optimizer.zero_grad(set_to_none=True)
        with autocast(c):
            loss, _ = loss_for_batch(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    for _ in range(2):
        iteration()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity, record_function
    # Shapes/stacks/memory tracing disabled to limit overhead, tensor retention and disk use.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(2):
            with record_function('nextbert_forward_backward_optimizer'):
                iteration()
        torch.cuda.synchronize()
    events = prof.key_averages()
    report = ('Operator diagnosis only: profiling adds overhead. One microbatch per optimizer step.\n\n'
              + events.table(sort_by='self_cpu_time_total', row_limit=35)+'\n\n'
              + events.table(sort_by='self_cuda_time_total', row_limit=35))
    (output/f'{variant}-operators.txt').write_text(report)
    if trace:
        prof.export_chrome_trace(str(output/f'{variant}-trace.json'))
    print(report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', choices=['prepare', 'train', 'profile', 'summary'], default='prepare')
    p.add_argument('--source', default='data/wiki-starter')
    p.add_argument('--output', default='data/wiki-long-check')
    p.add_argument('--per-language', type=int, default=4)
    p.add_argument('--variant', choices=['B0', 'G2'], default='G2')
    p.add_argument('--trace', action='store_true')
    args = p.parse_args()
    if args.per_language < 1:
        p.error('per-language must be positive')
    output = Path(args.output)
    if args.action in {'prepare', 'train'}:
        manifest = prepare(Path(args.source), output, args.per_language)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if args.action == 'train':
        # Separate processes release the complete CUDA context between variants.
        for variant in ('B0', 'G2'):
            subprocess.run([sys.executable, '-m', 'nextbert.cli', 'train',
                            '--config', str(output/f'{variant}.json')], check=True)
        summarize(output)
    elif args.action == 'summary':
        summarize(output)
    elif args.action == 'profile':
        profile_run(output, args.variant, args.trace)


if __name__ == '__main__':
    main()
