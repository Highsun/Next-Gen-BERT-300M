#!/usr/bin/env python3
"""Create controlled classification probes on the CUDA host, not during authoring.

This is a starter synthetic probe, not a substitute for human-reviewed data.
Each base group keeps position/counterfactual variants in the same split.
"""
import argparse
import json
import random
from pathlib import Path
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--model', default='jhu-clsp/mmBERT-base')
parser.add_argument('--revision', required=True, help='Use the experiment immutable SHA')
parser.add_argument('--count', type=int, default=600, help='Base samples per language')
parser.add_argument('--seed', type=int, default=17)
args = parser.parse_args()
tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
rng = random.Random(args.seed)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    raise FileExistsError(output)
with output.open('w') as handle:
    for language in ('zh', 'en'):
        for i in range(args.count):
            person = f'Entity{i:05d}'
            city = f'Place{rng.randrange(100000):05d}'
            group = f'{language}-{i}'
            question = (f'判断命题：{person}的登记地点是{city}。' if language == 'zh'
                        else f'Is it true that {person} is registered at {city}? ')
            for label, condition in ((1, 'supports'), (0, 'counterfactual'), (2, 'missing')):
                evidence = (f'记录明确指出：{person}的登记地点是{city}。' if language == 'zh'
                            else f'The official record states: {person} is registered at {city}. ')
                if label == 0:
                    evidence = evidence.replace(city, city+'Different')
                if label == 2:
                    evidence = '没有相关记录。' if language == 'zh' else 'No relevant record is available. '
                distractor = ('这是一条与登记地点无关的背景资料。' if language == 'zh'
                              else 'This is background material unrelated to the registration. ')
                for length in (512, 2048, 4096, 8192):
                    budget = length-len(tok(question+evidence, add_special_tokens=True)['input_ids'])-8
                    filler_ids = tok(distractor*(length//4), add_special_tokens=False)['input_ids'][:max(0, budget)]
                    for position, ratio in (('start', 0.), ('middle', .5), ('end', 1.)):
                        split = int(len(filler_ids)*ratio)
                        text = question+tok.decode(filler_ids[:split])+evidence+tok.decode(filler_ids[split:])
                        actual = len(tok(text, add_special_tokens=True)['input_ids'])
                        if actual > length:
                            raise ValueError('Retokenization exceeded length; reduce filler budget')
                        row = {'id': f'{group}-{condition}-{length}-{position}', 'group_id': group,
                               'base_id': group, 'language': language, 'label': label, 'text': text,
                               'condition': condition, 'position': position, 'length': actual,
                               'length_target': length,
                               'note': 'Single-evidence starter probe; no claim of compositional reasoning.'}
                        handle.write(json.dumps(row, ensure_ascii=False)+'\n')
