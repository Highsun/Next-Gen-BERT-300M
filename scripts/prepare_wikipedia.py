#!/usr/bin/env python3
"""Stream a bounded bilingual starter corpus; pin sources and share exact run configs.

Execute on AutoDL only. This is a startup corpus, not a decontaminated benchmark release.
Wikipedia licensing and article provenance are retained in each row.
"""
import argparse
import hashlib
import json
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', default='data/wiki-starter')
parser.add_argument('--tokens-per-language', type=int, default=2000000)
parser.add_argument('--model', default='jhu-clsp/mmBERT-base')
parser.add_argument('--model-revision', default='main')
parser.add_argument('--dataset-revision', default='main')
parser.add_argument('--seed', type=int, default=17)
parser.add_argument('--dev-percent', type=int, default=1)
args = parser.parse_args()
if args.tokens_per_language < 10000:
    parser.error('Use at least 10000 tokens per language to get nonempty splits')
if not 1 <= args.dev_percent <= 50:
    parser.error('dev-percent must be between 1 and 50')
output = Path(args.output_dir)
if output.exists():
    raise FileExistsError('Use a fresh directory, never overwrite a frozen corpus')
model_sha = (args.model_revision if len(args.model_revision) == 40 else
             HfApi().model_info(args.model, revision=args.model_revision).sha)
dataset_sha = (args.dataset_revision if len(args.dataset_revision) == 40 else
               HfApi().dataset_info('wikimedia/wikipedia', revision=args.dataset_revision).sha)
try:
    tok = AutoTokenizer.from_pretrained(args.model, revision=model_sha, use_fast=True, local_files_only=True)
except OSError:
    tok = AutoTokenizer.from_pretrained(args.model, revision=model_sha, use_fast=True)
output.mkdir(parents=True)
handles = {name: (output/f'{name}.jsonl').open('w', encoding='utf-8') for name in ('train', 'dev')}
counts = {lang: {'tokens': 0, 'documents': 0, 'train': 0, 'dev': 0} for lang in ('zh', 'en')}
try:
    streams = {lang: iter(load_dataset('wikimedia/wikipedia', f'20231101.{lang}',
                    revision=dataset_sha, split='train', streaming=True)) for lang in counts}
    seen = set()
    while any(v['tokens'] < args.tokens_per_language for v in counts.values()):
        for lang, stream in streams.items():
            if counts[lang]['tokens'] >= args.tokens_per_language:
                continue
            try:
                article = next(stream)
            except StopIteration as exc:
                raise RuntimeError(f'{lang} source exhausted before token target') from exc
            text = article['text'][:200000]
            encoded = tok(text, add_special_tokens=False, truncation=True, max_length=8190,
                          return_offsets_mapping=True)
            if len(encoded['input_ids']) < 128:
                continue
            text = text[:encoded['offset_mapping'][-1][1]]
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            doc_id = f"wiki-{lang}-{article['id']}"
            split_hash = hashlib.sha256(doc_id.encode()).hexdigest()
            split = 'dev' if int(split_hash[:8], 16) % 100 < args.dev_percent else 'train'
            row = {'id': doc_id, 'group_id': doc_id, 'language': lang, 'text': text,
                   'source': article.get('url'), 'source_revision': dataset_sha,
                   'content_sha256': digest,
                   'license_reference': 'https://huggingface.co/datasets/wikimedia/wikipedia'}
            handles[split].write(json.dumps(row, ensure_ascii=False)+'\n')
            counts[lang]['tokens'] += len(encoded['input_ids'])
            counts[lang]['documents'] += 1
            counts[lang][split] += 1
finally:
    for handle in handles.values():
        handle.close()
if any(counts[lang][split] == 0 for lang in counts for split in ('train', 'dev')):
    raise RuntimeError('Empty language/split: increase target and use a fresh directory')
manifest = {'model_name': args.model, 'model_revision': model_sha,
            'dataset_revision': dataset_sha, 'counts': counts, 'seed': args.seed,
            'sampling': f'Deterministic source order; interleave languages; exact dedup; doc hash {100-args.dev_percent}/{args.dev_percent}.',
            'limitations': 'First contiguous <=8190-token article window; no benchmark decontamination or near dedup.',
            'files': {name: hashlib.sha256((output/f'{name}.jsonl').read_bytes()).hexdigest()
                      for name in handles}}
(output/'manifest.json').write_text(json.dumps(manifest, indent=2))
for variant in ('B0', 'L3', 'M1', 'M2', 'G2', 'C2'):
    for smoke in (True, False):
        tag = f'{variant}-smoke' if smoke else f'{variant}-starter'
        config = {'model_name': args.model, 'revision': model_sha, 'variant': variant,
                  'task': 'mlm', 'seed': args.seed, 'lora_rank': 16, 'precision': 'bf16',
                  'max_length': 512 if smoke else 8192,
                  'max_steps': 20 if smoke else 1000, 'accumulation': 2 if smoke else 8,
                  'save_every': 10 if smoke else 100, 'keep_checkpoints': 2,
                  'log_every': 1 if smoke else 10, 'length_curriculum': not smoke,
                  'train_file': str(output/'train.jsonl'), 'output_dir': f'runs/{tag}'}
        (output/f'{tag}.json').write_text(json.dumps(config, indent=2))
print(json.dumps(manifest, indent=2))
print('Wrote pinned starter/smoke configs. Smoke success is not an ablation result.')
