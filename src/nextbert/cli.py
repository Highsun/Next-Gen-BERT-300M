"""CLI imports CUDA/model dependencies only for commands that need them."""
import argparse
from pathlib import Path
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(prog='nextbert')
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--config', required=True)
    train.add_argument('--resume')
    train.add_argument('--init')
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--data', required=True)
    evaluate.add_argument('--output', required=True)
    search = commands.add_parser('search')
    for arg in ('checkpoint', 'queries', 'corpus', 'qrels', 'output'):
        search.add_argument('--'+arg, required=True)
    search.add_argument('--candidates')
    search.add_argument('--top-k', type=int, default=100)
    pilot = commands.add_parser('pilot')
    pilot.add_argument('--config', required=True)
    pilot.add_argument('--output', required=True)
    pilot.add_argument('--lengths', default='512,2048,4096,8192')
    pilot.add_argument('--iterations', type=int, default=10)
    pilot.add_argument('--backward', action='store_true')
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--input', required=True)
    prepare.add_argument('--output-dir', required=True)
    prepare.add_argument('--task', required=True, choices=['mlm','classification','ner','qa','retrieval','rerank'])
    prepare.add_argument('--train-percent', type=int, default=80)
    prepare.add_argument('--dev-percent', type=int, default=10)
    args = parser.parse_args()
    if args.command == 'train':
        from .train import train
        train(args.config, args.resume, args.init)
    elif args.command == 'evaluate':
        from .evaluate import evaluate
        evaluate(args.checkpoint, args.data, args.output)
    elif args.command == 'search':
        from .evaluate import search
        if args.top_k < 1:
            parser.error('top-k must be positive')
        search(args.checkpoint, args.queries, args.corpus, args.qrels, args.output, args.candidates, args.top_k)
    elif args.command == 'pilot':
        from .pilot import pilot
        lengths = [int(x) for x in args.lengths.split(',')]
        if args.iterations < 1 or any(x < 8 or x > 8192 for x in lengths):
            parser.error('Invalid pilot iterations/lengths')
        pilot(args.config, args.output, lengths, args.iterations, args.backward)
    else:
        if not 0 < args.train_percent <= 100 or not 0 <= args.dev_percent <= 100-args.train_percent:
            parser.error('Invalid split percentages')
        from .data import rows, validate_row, file_hash
        output = Path(args.output_dir)
        if output.exists():
            raise FileExistsError('Use a new output directory; never overwrite a frozen split')
        output.mkdir(parents=True)
        seen, texts, count = set(), set(), {'train': 0, 'dev': 0, 'test': 0}
        handles = {name: (output/(name+'.jsonl')).open('w') for name in count}
        try:
            for row in rows(args.input):
                validate_row(row, args.task)
                if row['id'] in seen:
                    raise ValueError('Duplicate id')
                seen.add(row['id'])
                # Group translations, near-duplicate clusters, QA windows by upstream group_id.
                group = str(row.get('group_id', row['id']))
                split = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
                split = 'train' if split < args.train_percent else 'dev' if split < args.train_percent+args.dev_percent else 'test'
                text_hash = hashlib.sha256(row.get('text', row.get('query', '')).encode()).hexdigest()
                if text_hash in texts and args.task in {'mlm', 'classification'}:
                    continue
                texts.add(text_hash)
                handles[split].write(json.dumps(row, ensure_ascii=False)+'\n')
                count[split] += 1
        finally:
            for handle in handles.values():
                handle.close()
        (output/'manifest.json').write_text(json.dumps({'counts': count, 'source_sha256': file_hash(args.input),
            'split_rule': f'group SHA256 {args.train_percent}/{args.dev_percent}/{100-args.train_percent-args.dev_percent}; exact dedup only; near-duplicate group_id supplied upstream',
            'splits': {name: file_hash(output/(name+'.jsonl')) for name in count}}, indent=2))


if __name__ == '__main__':
    main()
