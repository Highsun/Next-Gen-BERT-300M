"""CUDA-only correctness/performance entry point, to be run on AutoDL."""
from pathlib import Path
import json
import time
import torch
from .config import Config
from .data import Collator, to_device
from .train import build, autocast, loss_for_batch


def pilot(config_path, output, lengths, iterations=10, backward=False):
    c = Config.load(config_path)
    if c.task != 'mlm':
        raise ValueError("Pilot uses MLM config; task-specific performance must be measured separately")
    model, tok = build(c)
    model.train(backward)
    collate = Collator(tok, c)
    records = []
    for length in lengths:
        # Actual token length is checked; synthetic repeated text is not a quality benchmark.
        c.max_length = length
        example = {'id': 'pilot', 'text': ('这是用于性能检查的句子。 A profiling sentence with numbers 123. '*length)}
        batch = to_device(collate([example], 0, length), 'cuda')
        n = int(batch['attention_mask'].sum())
        if n != length:
            raise ValueError("Pilot did not reach requested token length")
        def iteration():
            model.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(backward), autocast(c):
                loss, _ = loss_for_batch(model, batch)
            if backward:
                loss.backward()
        try:
            for _ in range(3):
                iteration()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                iteration()
                torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
            values = torch.tensor(times)
            records.append({'length': n, 'backward': backward, 'p50_seconds': float(values.quantile(.5)),
                            'p95_seconds': float(values.quantile(.95)), 'tokens_per_second': n/(sum(times)/len(times)),
                            'allocated_bytes': torch.cuda.max_memory_allocated(),
                            'reserved_bytes': torch.cuda.max_memory_reserved()})
        except torch.cuda.OutOfMemoryError:
            records.append({'length': n, 'oom': True})
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps({'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                                      'cuda': torch.version.cuda, 'config': c.to_dict(), 'records': records}, indent=2))
