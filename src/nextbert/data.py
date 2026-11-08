"""Task-neutral JSONL contract; deterministic corruption, char-aligned extraction."""
from pathlib import Path
import hashlib
import json
import random
import torch


def rows(path):
    with open(path, encoding="utf-8") as handle:
        for line, text in enumerate(handle, 1):
            if text.strip():
                value = json.loads(text)
                if not isinstance(value, dict) or 'id' not in value:
                    raise ValueError(f"{path}:{line}: requires object with id")
                yield value


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for part in iter(lambda: handle.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def validate_row(row, task):
    required = {'mlm': ['text'], 'classification': ['text', 'label'],
                'ner': ['text', 'entities'], 'qa': ['query', 'text', 'answers'],
                'retrieval': ['query', 'positive', 'negatives'],
                'rerank': ['query', 'text', 'label']}[task]
    for key in required:
        if key not in row:
            raise ValueError(f"{row['id']}: missing {key}")
    for key in ('text', 'query', 'positive'):
        if key in row and (not isinstance(row[key], str) or not row[key].strip()):
            raise ValueError(f"{row['id']}: empty/non-string {key}")
    if task == 'retrieval' and (not row['negatives'] or any(not isinstance(x, str) or not x.strip() for x in row['negatives'])):
        raise ValueError("Retrieval requires nonempty string negatives")
    if task == 'rerank' and row['label'] not in (0, 1):
        raise ValueError('Rerank training label must be binary; use relevance for graded judgments')
    if task == 'retrieval' and row['positive'] in row['negatives']:
        raise ValueError('Known positive present in negatives')
    if task == 'ner':
        previous = 0
        for entity in sorted(row['entities'], key=lambda x: x['start']):
            a, b = entity['start'], entity['end']
            if not previous <= a < b <= len(row['text']):
                raise ValueError("NER requires non-overlapping valid char spans")
            previous = b
    if task == 'qa':
        for answer in row['answers']:
            a, text = answer['start'], answer['text']
            if not text or a < 0 or row['text'][a:a+len(text)] != text:
                raise ValueError("QA answer does not match original text")


class Collator:
    def __init__(self, tokenizer, config):
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer with char offsets is required")
        self.tok, self.c = tokenizer, config
        # Tokenizer is immutable during a run; avoid repeated special-token conversion per replacement.
        self.vocab_size = len(tokenizer)
        self.special_ids = frozenset(tokenizer.all_special_ids)
        self.mask_id = tokenizer.mask_token_id

    def plain(self, texts, max_length=None):
        enc = self.tok(texts, padding=True, truncation=True, max_length=max_length or self.c.max_length,
                       return_special_tokens_mask=True, return_tensors='pt')
        special = enc.pop('special_tokens_mask').bool()
        enc.pop('token_type_ids', None)
        enc['content_mask'] = enc['attention_mask'].bool() & ~special
        if not bool(enc['content_mask'].any(-1).all()):
            raise ValueError("Empty tokenized content")
        return dict(enc)

    def __call__(self, examples, step=0, max_length=None):
        task = self.c.task
        for row in examples:
            validate_row(row, task)
        if task == 'retrieval':
            q = self.plain([x['query'] for x in examples], self.c.query_length+2)
            docs = []
            groups = []
            for ex in examples:
                groups.append(len(docs))
                docs.extend([ex['positive'], *ex['negatives']])
            return {'queries': q, 'documents': self.plain(docs, max_length),
                    'targets': torch.tensor(groups)}
        if task in {'qa', 'rerank'}:
            # Truncate query BEFORE pair encoding to preserve document budget.
            queries = [self.tok.decode(self.tok(x['query'], add_special_tokens=False)['input_ids'][:self.c.query_length],
                                       skip_special_tokens=True) for x in examples]
            enc = self.tok(queries, [x['text'] for x in examples], truncation='only_second', padding=True,
                           max_length=max_length or self.c.max_length, return_offsets_mapping=True,
                           return_special_tokens_mask=True, return_tensors='pt')
            seq = [enc.sequence_ids(i) for i in range(len(examples))]
            offsets = enc.pop('offset_mapping')
            enc.pop('special_tokens_mask')
            enc.pop('token_type_ids', None)
            enc['content_mask'] = torch.tensor([[v == 1 for v in s] for s in seq])
            enc['query_mask'] = torch.tensor([[v == 0 for v in s] for s in seq])
            if task == 'rerank':
                enc['labels'] = torch.tensor([float(x['label']) for x in examples])
            else:
                starts, ends = [], []
                for i, ex in enumerate(examples):
                    start, end = 0, 0
                    if ex['answers']:
                        answer = ex['answers'][0]
                        a, b = answer['start'], answer['start']+len(answer['text'])
                        si = [j for j, ((left, right), s) in enumerate(zip(offsets[i].tolist(), seq[i]))
                              if s == 1 and left <= a < right]
                        ei = [j for j, ((left, right), s) in enumerate(zip(offsets[i].tolist(), seq[i]))
                              if s == 1 and left < b <= right]
                        # Truncated positive answers are ignored, never mislabeled as null.
                        start, end = (si[0], ei[0]) if si and ei else (-100, -100)
                    starts.append(start)
                    ends.append(end)
                enc['start'], enc['end'] = torch.tensor(starts), torch.tensor(ends)
                enc['offsets'] = offsets
            return dict(enc)
        if task == 'ner':
            enc = self.tok([x['text'] for x in examples], truncation=True, padding=True,
                           max_length=max_length or self.c.max_length, return_offsets_mapping=True,
                           return_special_tokens_mask=True, return_tensors='pt')
            offsets = enc.pop('offset_mapping')
            special = enc.pop('special_tokens_mask').bool()
            enc.pop('token_type_ids', None)
            enc['content_mask'] = enc['attention_mask'].bool() & ~special
            labels = torch.full_like(enc['input_ids'], -100)
            names = {name: i for i, name in enumerate(self.c.label_names)}
            if 'O' not in names:
                raise ValueError("NER label_names must include O and BIO labels")
            for i, ex in enumerate(examples):
                for j, (a, b) in enumerate(offsets[i].tolist()):
                    if not enc['content_mask'][i, j]:
                        continue
                    labels[i, j] = names['O']
                    for entity in ex['entities']:
                        if a < entity['end'] and b > entity['start']:
                            prefix = 'B-' if a <= entity['start'] else 'I-'
                            labels[i, j] = names[prefix+entity['type']]
                            break
            enc['labels'], enc['offsets'] = labels, offsets
            return dict(enc)
        enc = self.plain([x['text'] for x in examples], max_length)
        if task == 'classification':
            enc['labels'] = torch.tensor([int(x['label']) for x in examples])
            return enc
        labels = torch.full_like(enc['input_ids'], -100)
        for i, ex in enumerate(examples):
            rng = random.Random(f"{self.c.seed}:{step}:{ex['id']}")
            available = enc['content_mask'][i].nonzero(as_tuple=True)[0].tolist()
            target = max(1, round(len(available)*self.c.mask_probability))
            selected = set()
            span = rng.random() < self.c.span_batch_probability
            while len(selected) < target:
                pos = rng.randrange(len(available))
                length = 1
                if span:
                    while length < 10 and rng.random() > 1/3:
                        length += 1
                group = available[pos:pos+min(length, target-len(selected))]
                operation = rng.random()
                for j in group:
                    if j in selected:
                        continue
                    selected.add(j)
                    labels[i, j] = enc['input_ids'][i, j]
                    if operation < 0.8:
                        enc['input_ids'][i, j] = self.mask_id
                    elif operation < 0.9:
                        replacement = rng.randrange(self.vocab_size)
                        while replacement in self.special_ids:
                            replacement = rng.randrange(self.vocab_size)
                        enc['input_ids'][i, j] = replacement
        enc['labels'] = labels
        return enc


def to_device(batch, device):
    return {key: to_device(value, device) if isinstance(value, dict)
            else value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
