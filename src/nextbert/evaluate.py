"""Task evaluation, cached retrieval, and paired diagnostics. No benchmark downloads."""
from pathlib import Path
from collections import Counter, defaultdict
import json
import math
import re
import string
import time
import numpy as np
import torch
from .data import Collator, file_hash, rows, to_device
from .train import autocast, load_weights


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def bio_spans(labels, offsets):
    spans = []
    active = None
    for label, (a, b) in zip(labels, offsets):
        kind = label[2:] if label.startswith(('B-', 'I-')) else None
        if active is not None and (label.startswith('B-') or kind != active[2]):
            spans.append(tuple(active))
            active = None
        if kind is not None:
            if active is None:
                active = [a, b, kind]
            else:
                active[1] = b
    if active is not None:
        spans.append(tuple(active))
    return spans


def normalize(text):
    text = text.lower()
    text = ''.join(x for x in text if x not in string.punctuation+'，。！？；：“”‘’（）【】、')
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_metrics(prediction, answers, language):
    def units(text):
        text = normalize(text)
        return list(text.replace(' ', '')) if language.startswith('zh') else text.split()
    em, f1 = 0., 0.
    for answer in answers or ['']:
        p, g = units(prediction), units(answer)
        common = sum((Counter(p) & Counter(g)).values())
        score = 2*common/(len(p)+len(g)) if p and g else float(p == g)
        em, f1 = max(em, float(normalize(prediction) == normalize(answer))), max(f1, score)
    return em, f1


def decode_qa(logits, offsets, content, text, max_span=64):
    start, end = logits[:, 0], logits[:, 1]
    positions = content.nonzero(as_tuple=True)[0].tolist()
    candidates = [(float(start[a]+end[b]), a, b) for a in sorted(positions, key=lambda x: float(start[x]), reverse=True)[:20]
                  for b in sorted(positions, key=lambda x: float(end[x]), reverse=True)[:20]
                  if a <= b < a+max_span]
    if not candidates:
        return ''
    score, a, b = max(candidates)
    if score <= float(start[0]+end[0]):
        return ''
    return text[int(offsets[a, 0]):int(offsets[b, 1])]


def rank_metrics(ranked, qrels, cutoffs=(10, 100)):
    relevant = {d for d, grade in qrels.items() if grade > 0}
    if not relevant:
        raise ValueError("Query has no positive qrels; filter explicitly before evaluation")
    result = {}
    for k in cutoffs:
        grades = [qrels.get(d, 0) for d in ranked[:k]]
        dcg = sum((2**g-1)/math.log2(i+2) for i, g in enumerate(grades))
        ideal = sorted(qrels.values(), reverse=True)[:k]
        idcg = sum((2**g-1)/math.log2(i+2) for i, g in enumerate(ideal))
        result[f'ndcg@{k}'] = dcg/idcg if idcg else 0.
        result[f'recall@{k}'] = len(set(ranked[:k]) & relevant)/len(relevant)
        result[f'mrr@{k}'] = next((1/(i+1) for i, d in enumerate(ranked[:k]) if d in relevant), 0.)
    return result


@torch.inference_mode()
def evaluate(checkpoint, data_path, output):
    model, tok, c, _ = load_weights(checkpoint)
    if c.task == 'retrieval':
        raise ValueError("Use search for corpus-level retrieval evaluation")
    model.eval()
    collate = Collator(tok, c)
    records, losses, golds, preds = [], [], [], []
    ner_tp, ner_fp, ner_fn = 0, 0, 0
    ranking = defaultdict(list)
    for ex in rows(data_path):
        batch = to_device(collate([ex], step=0), 'cuda')
        if c.task == 'qa' and int(batch['start'][0]) == -100:
            raise ValueError(f"QA answer truncated for {ex['id']}; use prepared windows")
        with autocast(c):
            loss, logits = model(batch)
        losses.append(float(loss))
        record = {'id': ex['id'], 'language': ex.get('language', 'unknown')}
        for field in ('base_id', 'condition', 'distance', 'position', 'length'):
            if field in ex:
                record[field] = ex[field]
        if c.task == 'mlm':
            record['loss'] = float(loss)
        elif c.task == 'classification':
            pred = int(logits.argmax(-1)[0])
            record.update(prediction=pred, gold=ex['label'], correct=pred == ex['label'])
            golds.append(ex['label'])
            preds.append(pred)
        elif c.task == 'ner':
            valid = batch['labels'][0] != -100
            gold = [c.label_names[x] for x in batch['labels'][0, valid].tolist()]
            pred = [c.label_names[x] for x in logits[0, valid].argmax(-1).tolist()]
            golds.append(gold)
            preds.append(pred)
            offsets = batch['offsets'][0, valid].tolist()
            spans = bio_spans(pred, offsets)
            predicted = {(a, b, kind) for a, b, kind in spans}
            # Fail on truncation instead of silently dropping difficult gold entities.
            last = max(b for _, b in offsets)
            if any(entity['end'] > last for entity in ex['entities']):
                raise ValueError('NER gold span truncated; prepare explicit windows')
            expected = {(e['start'], e['end'], e['type']) for e in ex['entities']}
            ner_tp += len(predicted & expected)
            ner_fp += len(predicted - expected)
            ner_fn += len(expected - predicted)
            record.update(prediction=pred, gold=gold, offsets=offsets, predicted_spans=spans)
        elif c.task == 'qa':
            pred = decode_qa(logits[0], batch['offsets'][0], batch['content_mask'][0], ex['text'])
            em, f1 = answer_metrics(pred, [a['text'] for a in ex['answers']], ex.get('language', 'en'))
            record.update(prediction=pred, em=em, f1=f1, correct=bool(em))
        else:
            if 'query_id' not in ex or 'doc_id' not in ex:
                raise ValueError("Rerank evaluation requires query_id/doc_id and graded relevance")
            score = float(logits[0])
            ranking[ex['query_id']].append((ex['doc_id'], score, ex.get('relevance', ex['label'])))
            record.update(score=score, query_id=ex['query_id'], doc_id=ex['doc_id'])
        records.append(record)
    if not records:
        raise ValueError("Empty evaluation file")
    metrics = {'loss': float(np.mean(losses)), 'count': len(records), 'data_sha256': file_hash(data_path)}
    if c.task == 'classification':
        metrics['accuracy'] = sum(p == g for p, g in zip(preds, golds))/len(golds)
        f1s = []
        for label in range(c.num_labels):
            tp = sum(p == g == label for p, g in zip(preds, golds))
            fp = sum(p == label and g != label for p, g in zip(preds, golds))
            fn = sum(p != label and g == label for p, g in zip(preds, golds))
            f1s.append(2*tp/max(1, 2*tp+fp+fn))
        metrics['macro_f1'] = float(np.mean(f1s))
    elif c.task == 'ner':
        from seqeval.metrics import f1_score
        metrics['subtoken_bio_f1'] = f1_score(golds, preds)
        metrics['character_entity_f1'] = 2*ner_tp/max(1, 2*ner_tp+ner_fp+ner_fn)
        metrics['note'] = 'Exact original-character entity spans; official benchmark scorer still required for final comparison.'
    elif c.task == 'qa':
        metrics.update(em=float(np.mean([r['em'] for r in records])), f1=float(np.mean([r['f1'] for r in records])))
        metrics['note'] = 'Generic normalized EN-word/ZH-character metrics; not an official CMRC/SQuAD scorer.'
    elif c.task == 'rerank':
        per_query = []
        for query_id, values in ranking.items():
            ranked = [d for d, _, _ in sorted(values, key=lambda x: (-x[1], x[0]))]
            qm = rank_metrics(ranked, {d: int(label) for d, _, label in values})
            per_query.append({'id': query_id, **qm})
        metrics.update({k: float(np.mean([r[k] for r in per_query])) for k in per_query[0] if k != 'id'})
        metrics['note'] = 'Candidate-only qrels; use search --qrels full judgments for corpus recall.'
        write_json(str(output)+'.queries.json', per_query)
    write_json(output, metrics)
    Path(str(output)+'.predictions.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in records))


@torch.inference_mode()
def search(checkpoint, queries_path, corpus_path, qrels_path, output, candidates=None, top_k=100):
    model, tok, c, _ = load_weights(checkpoint)
    model.eval()
    collate = Collator(tok, c)
    corpus = list(rows(corpus_path))
    ids = [x['id'] for x in corpus]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Corpus must be nonempty and IDs unique")
    # Cache stays on host; candidate views are moved individually to CUDA.
    cache = []
    start = time.monotonic()
    for doc in corpus:
        with autocast(c):
            views = model.encode_views(to_device(collate.plain([doc['text']]), 'cuda'))
        cache.append((views[0].cpu(), [x.cpu() for x in views[3]]))
    index_seconds = time.monotonic()-start
    dense = torch.cat([v[0] for v in cache])
    judgments = defaultdict(dict)
    for row in rows(qrels_path):
        judgments[row['query_id']][row['doc_id']] = int(row['relevance'])
    fixed = {x['id']: x['doc_ids'] for x in rows(candidates)} if candidates else None
    all_results, trec = [], []
    lookup = {doc_id: i for i, doc_id in enumerate(ids)}
    for query in rows(queries_path):
        if query['id'] not in judgments:
            raise ValueError("Missing qrels for query")
        with autocast(c):
            q = model.encode_views(to_device(collate.plain([query['text']], c.query_length+2), 'cuda'))
        base_scores = (q[0].cpu() @ dense.T)[0]
        indices = [lookup[d] for d in fixed[query['id']]] if fixed is not None else base_scores.topk(min(top_k, len(ids))).indices.tolist()
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate candidate documents")
        scored = []
        for i in indices:
            dg, blocks = cache[i]
            doc_views = (dg.cuda(), None, None, [x.cuda() for x in blocks])
            scored.append((ids[i], float(model.score_views(q, doc_views)[0, 0])))
        ranked = sorted(scored, key=lambda x: (-x[1], x[0]))
        result = {'id': query['id'], **rank_metrics([d for d, _ in ranked], judgments[query['id']])}
        result['candidate_recall'] = rank_metrics([ids[i] for i in indices], judgments[query['id']], (len(indices),))[f'recall@{len(indices)}']
        all_results.append(result)
        trec.extend(f"{query['id']} Q0 {d} {rank} {score:.8f} nextbert\n" for rank, (d, score) in enumerate(ranked, 1))
    if not all_results:
        raise ValueError("No queries")
    write_json(output, {'metrics': {k: float(np.mean([r[k] for r in all_results])) for k in all_results[0] if k != 'id'},
                        'per_query': all_results, 'index_seconds': index_seconds,
                        'corpus_sha256': file_hash(corpus_path), 'queries_sha256': file_hash(queries_path),
                        'qrels_sha256': file_hash(qrels_path), 'search_backend': 'exact CPU dense + CUDA candidate block scoring',
                        'note': 'Research reference search, not a production ANN throughput benchmark.'})
    Path(str(output)+'.trec').write_text(''.join(trec))
