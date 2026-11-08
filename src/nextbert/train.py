"""Single CUDA GPU trainer with reproducible sample cursor and full resume state."""
from contextlib import nullcontext
from pathlib import Path
import json
import math
import os
import random
import shutil
import time
import numpy as np
import torch
from transformers import AutoTokenizer
from .config import Config
from .data import Collator, file_hash, rows, to_device
from .model import Encoder


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(config):
    return torch.autocast('cuda', dtype=torch.bfloat16) if config.precision == 'bf16' else nullcontext()


def build(config):
    if not torch.cuda.is_available():
        raise RuntimeError("Training/inference requires Ubuntu + CUDA; no silent CPU fallback")
    if config.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16; explicitly select fp32")
    seed_all(config.seed)
    from huggingface_hub import HfApi
    if len(config.revision) != 40:
        config.revision = HfApi().model_info(config.model_name, revision=config.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision, use_fast=True)
    tokenizer.padding_side = 'right'
    model = Encoder(config).cuda()
    return model, tokenizer


def save_checkpoint(path, model, tokenizer, config, optimizer, state):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    names = {name for name, param in model.named_parameters() if param.requires_grad}
    weights = model.state_dict()
    compact = bool(config.lora_rank)
    if compact:
        weights = {name: value for name, value in weights.items() if name in names}
    payload = {'model': weights, 'checkpoint_format': 'trainable-v1' if compact else 'full-v1',
               'optimizer': optimizer.state_dict(), 'state': state,
               'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
               'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
               'config': config.to_dict()}
    temp = path/'checkpoint.tmp'
    torch.save(payload, temp)
    os.replace(temp, path/'checkpoint.pt')
    (path/'config.json').write_text(json.dumps(config.to_dict(), indent=2))
    tokenizer.save_pretrained(path/'tokenizer')


def restore_state(model, payload, backbone_only=False):
    """Compact checkpoints require the same immutable base revision and LoRA rank."""
    compact = payload.get('checkpoint_format') == 'trainable-v1'
    if compact and not model.exp.lora_rank:
        raise ValueError('Cannot load LoRA-only checkpoint into full-parameter model')
    target = model.state_dict()
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    expected = trainable if compact else set(target)
    source = payload['model']
    if backbone_only:
        source = {k: v for k, v in source.items() if k.startswith(('lm.', 'branches.'))}
        expected = {k for k in expected if k.startswith(('lm.', 'branches.'))}
    if set(source) != expected or any(k not in target or target[k].shape != v.shape for k, v in source.items()):
        raise ValueError('Checkpoint architecture, task heads, or trainable parameter set mismatch')
    if compact and (payload['config']['model_name'] != model.exp.model_name
                    or payload['config']['revision'] != model.exp.revision):
        raise ValueError('Compact checkpoint base model/revision mismatch')
    model.load_state_dict(source, strict=not compact and not backbone_only)


def prune_checkpoints(output, keep):
    """Called only after a new checkpoint has been completely written."""
    checkpoints = sorted(p for p in Path(output).glob('step-*')
                         if p.is_dir() and p.name[5:].isdigit() and (p/'checkpoint.pt').is_file())
    for old in checkpoints[:-keep]:
        shutil.rmtree(old)


def load_weights(directory):
    # Only load trusted, self-produced checkpoints (optimizer/RNG require pickle).
    payload = torch.load(Path(directory)/'checkpoint.pt', map_location='cpu', weights_only=False)
    config = Config(**payload['config']).validate()
    model, tok = build(config)
    restore_state(model, payload)
    model.transition = 1.0
    return model, tok, config, payload


def loss_for_batch(model, batch):
    if model.exp.task != 'retrieval':
        loss, _ = model(batch)
        return loss, model.last_forward_flops
    q = model.encode_views(batch['queries'])
    qflops = model.last_forward_flops
    d = model.encode_views(batch['documents'])
    scores = model.score_views(q, d)/model.exp.temperature
    return torch.nn.functional.cross_entropy(scores, batch['targets']), qflops+model.last_forward_flops


def train(config_path, resume=None, init=None):
    c = Config.load(config_path)
    if resume and init:
        raise ValueError("Choose resume OR init")
    if not Path(c.train_file).is_file():
        raise FileNotFoundError(f'Training data not found: {c.train_file}; prepare data first')
    if not resume and (Path(c.output_dir)/'provenance.json').exists():
        raise FileExistsError('Output already contains a run; use --resume or a new output_dir')
    if resume:
        saved_config = Config.load(Path(resume)/'config.json')
        if c.revision == 'main':
            c.revision = saved_config.revision
    if init:
        payload = torch.load(Path(init)/'checkpoint.pt', map_location='cpu', weights_only=False)
        if c.revision == 'main':
            c.revision = payload['config']['revision']
    model, tok = build(c)
    if init:
        restore_state(model, payload, backbone_only=True)
        del payload
    examples = list(rows(c.train_file))
    if not examples:
        raise ValueError("Empty training file")
    digest = file_hash(c.train_file)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=c.learning_rate, weight_decay=c.weight_decay)
    state = {'step': 0, 'epoch': 0, 'cursor': 0, 'tokens': 0, 'estimated_flops': 0.,
             'elapsed_seconds': 0., 'length_tokens': {'512': 0, '2048': 0, '8192': 0}, 'data_sha256': digest}
    if resume:
        payload = torch.load(Path(resume)/'checkpoint.pt', map_location='cpu', weights_only=False)
        # Resume means exact config, including data and max_steps; use init for a new stage.
        if payload['config'] != c.to_dict() or payload['state']['data_sha256'] != digest:
            raise ValueError("Resume config or training data changed")
        restore_state(model, payload)
        optimizer.load_state_dict(payload['optimizer'])
        state = payload['state']
        random.setstate(payload['python_rng'])
        np.random.set_state(payload['numpy_rng'])
        torch.set_rng_state(payload['torch_rng'])
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    out = Path(c.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    provenance = {'config': c.to_dict(), 'data_sha256': digest,
                  'resolved_model_revision': getattr(model.lm.config, '_commit_hash', None),
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'gpu': torch.cuda.get_device_name(),
                  'parameters': sum(p.numel() for p in model.parameters()),
                  'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'length_note': 'Post-warmup token-deficit controller targets 30/30/40; clipping and finite-run deviations recorded.',
                  'flops_note': 'Analytical estimate only; not calibrated compute-equivalence evidence.'}
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    collate = Collator(tok, c)
    model.train()
    clock = time.monotonic()
    previous_elapsed = state['elapsed_seconds']
    order_epoch, order = None, []
    try:
        while state['step'] < c.max_steps:
            optimizer.zero_grad(set_to_none=True)
            progress = (state['estimated_flops']/c.max_estimated_flops if c.max_estimated_flops
                        else state['step']/c.max_steps)
            model.transition = min(1., progress/c.transition_fraction) if c.transition_fraction else 1.
            warmup = max(1, int(c.max_steps*c.warmup_fraction))
            multiplier = min((state['step']+1)/warmup,
                             max(0., (c.max_steps-state['step'])/max(1, c.max_steps-warmup)))
            for group in optimizer.param_groups:
                group['lr'] = c.learning_rate*multiplier
            loss_value = 0.
            for micro in range(c.accumulation):
                if state['cursor'] >= len(examples):
                    state['epoch'] += 1
                    state['cursor'] = 0
                if order_epoch != state['epoch']:
                    order = list(range(len(examples)))
                    random.Random(c.seed+state['epoch']).shuffle(order)
                    order_epoch = state['epoch']
                example = examples[order[state['cursor']]]
                state['cursor'] += 1
                length = c.max_length
                if c.length_curriculum and c.task == 'mlm':
                    if progress < .2:
                        length = 512
                    else:
                        targets = {'512': .3, '2048': .3, '8192': .4}
                        bucket = min(targets, key=lambda k: state['length_tokens'][k]/targets[k])
                        length = int(bucket)
                    length = min(length, c.max_length)
                batch = to_device(collate([example], state['step']*c.accumulation+micro, length), 'cuda')
                if c.task == 'qa' and int(batch['start'][0]) == -100:
                    raise ValueError(f"Positive QA answer truncated: {example['id']}; prepare answer-covering windows")
                with autocast(c):
                    loss, forward_flops = loss_for_batch(model, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss; checkpoint not advanced")
                (loss/c.accumulation).backward()
                loss_value += float(loss.detach())/c.accumulation
                parts = [batch['queries'], batch['documents']] if c.task == 'retrieval' else [batch]
                used_tokens = sum(int(x['attention_mask'].sum()) for x in parts)
                state['tokens'] += used_tokens
                if c.task == 'mlm' and progress >= .2:
                    state['length_tokens'][str(length)] = state['length_tokens'].get(str(length), 0)+used_tokens
                state['estimated_flops'] += forward_flops*(4 if c.checkpointing else 3)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            state['step'] += 1
            torch.cuda.synchronize()
            state['elapsed_seconds'] = previous_elapsed+time.monotonic()-clock
            record = {**state, 'loss': loss_value, 'transition': model.transition,
                      'lr': optimizer.param_groups[0]['lr'], 'max_memory_bytes': torch.cuda.max_memory_allocated()}
            with (out/'train.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            if state['step'] % c.log_every == 0:
                print(json.dumps(record), flush=True)
            if state['step'] % c.save_every == 0:
                save_checkpoint(out/f"step-{state['step']:06d}", model, tok, c, optimizer, state)
                prune_checkpoints(out, c.keep_checkpoints)
            if ((c.max_estimated_flops and state['estimated_flops'] >= c.max_estimated_flops)
                    or (c.max_seconds and state['elapsed_seconds'] >= c.max_seconds)):
                break
    except (KeyboardInterrupt, torch.cuda.OutOfMemoryError):
        # Do not write partially accumulated optimizer state as an exact-resume checkpoint.
        (out/'interrupted.txt').write_text('Resume the last completed step checkpoint.\n')
        raise
    save_checkpoint(out/'final', model, tok, c, optimizer, state)
