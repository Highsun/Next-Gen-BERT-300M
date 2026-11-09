#!/usr/bin/env python3
"""Generate and run a bounded, paired-data MLM screen, not an application-quality verdict."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', choices=['download', 'prepare', 'train', 'evaluate', 'summary'], required=True)
    p.add_argument('--data', default='data/wiki-phase1')
    p.add_argument('--output', default='outputs/stage1-s17')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--tokens-per-language', type=int, default=10000000)
    p.add_argument('--reference', default='data/wiki-starter/manifest.json')
    p.add_argument('--resume', action='store_true', help='Resume each arm from its most recent completed checkpoint')
    a = p.parse_args()
    if a.action == 'download':
        original = json.loads(Path(a.reference).read_text())
        subprocess.run([sys.executable, 'scripts/prepare_wikipedia.py', '--output-dir', a.data,
                        '--tokens-per-language', str(a.tokens_per_language), '--dev-percent', '10',
                        '--model', original['model_name'], '--model-revision', original['model_revision'],
                        '--dataset-revision', original['dataset_revision'], '--seed', str(a.seed)], check=True)
        return
    from nextbert.data import rows, file_hash
    source, output = Path(a.data), Path(a.output)
    manifest = json.loads((source/'manifest.json').read_text())
    for split in ('train', 'dev'):
        if file_hash(source/f'{split}.jsonl') != manifest['files'][split]:
            raise ValueError('Frozen corpus hash mismatch')
    if a.steps <= 0:
        p.error('steps must be positive')
    variants = ('B0', 'L3', 'G2')
    if a.action == 'prepare':
        if output.exists():
            raise FileExistsError('Choose a fresh stage output directory')
        output.mkdir(parents=True)
        language_rows = {lang: [] for lang in ('zh', 'en')}
        for row in rows(source/'dev.jsonl'):
            language_rows[row['language']].append(row)
        if any(len(v) < 100 for v in language_rows.values()):
            raise ValueError('Need at least 100 held-out documents per language; increase corpus size')
        for lang, values in language_rows.items():
            (output/f'dev.{lang}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in values))
        for variant in variants:
            config = {'model_name': manifest['model_name'], 'revision': manifest['model_revision'],
                      'variant': variant, 'task': 'mlm', 'seed': a.seed,
                      'lora_rank': 16, 'precision': 'bf16', 'max_length': 8192,
                      'max_steps': a.steps, 'accumulation': 8, 'learning_rate': 0.0001,
                      'length_curriculum': True, 'profile_pipeline': False,
                      'save_every': 250, 'keep_checkpoints': 2, 'log_every': 10,
                      'train_file': str(source/'train.jsonl'),
                      'output_dir': f'runs/{output.name}/{variant}'}
            write(output/f'{variant}.json', config)
        write(output/'stage-manifest.json', {
            'data': str(source), 'data_sha256': manifest['files'], 'model_revision': manifest['model_revision'],
            'seed': a.seed, 'steps': a.steps, 'variants': variants,
            'dev_counts': {k: len(v) for k, v in language_rows.items()},
            'dev_file_sha256': {lang: file_hash(output/f'dev.{lang}.jsonl') for lang in language_rows},
            'purpose': 'Same-data/exposure single-seed MLM screen; NOT equal calibrated FLOPs or downstream proof.'})
        print(f'Frozen configs in {output}. Run train and evaluate separately.', flush=True)
        return
    stage = json.loads((output/'stage-manifest.json').read_text())
    if stage['data_sha256'] != manifest['files'] or stage['seed'] != a.seed or stage['steps'] != a.steps:
        raise ValueError('Arguments do not match frozen stage; reuse the original arguments')
    for lang, digest in stage['dev_file_sha256'].items():
        if file_hash(output/f'dev.{lang}.jsonl') != digest:
            raise ValueError('Language dev set changed after freezing')
    for variant in variants:
        config_path = output/f'{variant}.json'
        config = json.loads(config_path.read_text())
        run = Path(config['output_dir'])
        if a.action == 'train':
            if (run/'final'/'checkpoint.pt').exists():
                history = [json.loads(line) for line in (run/'train.jsonl').read_text().splitlines() if line.strip()]
                if history and history[-1]['step'] == a.steps:
                    print(f'{variant} already completed; preserving results', flush=True)
                    continue
            command = [sys.executable, '-m', 'nextbert.cli', 'train', '--config', str(config_path)]
            if (run/'provenance.json').exists():
                if not a.resume:
                    raise ValueError(f'{variant} has an interrupted run; use --resume instead of overwriting')
                checkpoints = sorted(p for p in run.glob('step-*') if (p/'checkpoint.pt').is_file())
                if not checkpoints:
                    raise ValueError('No completed step checkpoint available; keep logs and choose a fresh run')
                # Remove only trailing log records not represented by the saved optimizer state.
                state = json.loads((checkpoints[-1]/'config.json').read_text())
                if state['max_steps'] != a.steps:
                    raise ValueError('Checkpoint schedule mismatch')
                saved_step = int(checkpoints[-1].name.split('-')[-1])
                log = run/'train.jsonl'
                backup = run/f'train-before-resume-{saved_step}.jsonl'
                if backup.exists():
                    raise FileExistsError('Resume backup already exists; inspect before another recovery')
                backup.write_bytes(log.read_bytes())
                records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
                log.write_text(''.join(json.dumps(r)+'\n' for r in records if r['step'] <= saved_step))
                command += ['--resume', str(checkpoints[-1])]
            print('Starting '+variant, flush=True)
            subprocess.run(command, check=True)
        elif a.action == 'evaluate':
            for lang in ('zh', 'en'):
                subprocess.run([sys.executable, '-m', 'nextbert.cli', 'evaluate',
                                '--checkpoint', str(run/'final'), '--data', str(output/f'dev.{lang}.jsonl'),
                                '--output', str(output/f'{variant}-dev-{lang}.json')], check=True)
    if a.action == 'summary':
        result = {'purpose': stage['purpose'], 'results': {}}
        for variant in variants:
            config = json.loads((output/f'{variant}.json').read_text())
            history = [json.loads(line) for line in (Path(config['output_dir'])/'train.jsonl').read_text().splitlines() if line.strip()]
            if history[-1]['step'] != a.steps:
                raise ValueError('Incomplete training')
            result['results'][variant] = {'training': history[-1], 'dev': {
                lang: json.loads((output/f'{variant}-dev-{lang}.json').read_text()) for lang in ('zh', 'en')}}
        write(output/'summary.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
