#!/usr/bin/env python3
"""Compile text without importing project code or writing __pycache__."""
from pathlib import Path
import ast
import datetime
import json
import tomllib

root = Path(__file__).resolve().parents[1]
files = sorted(root.glob('src/**/*.py')) + sorted(root.glob('scripts/*.py')) + sorted(root.glob('tests/*.py'))
for path in files:
    text = path.read_text()
    ast.parse(text, filename=str(path))
    compile(text, str(path), 'exec')
metadata = tomllib.loads((root/'pyproject.toml').read_text())
for path in root.glob('configs/*.json'):
    config = json.loads(path.read_text())
    assert config['variant'] in {'B0', 'L3', 'M1', 'M2', 'G2', 'C2'}, path
    assert config['max_length'] <= 8192, path
    assert config['task'] in {'mlm', 'classification', 'ner', 'qa', 'retrieval', 'rerank'}, path
report = {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'python_files': [str(p.relative_to(root)) for p in files],
          'checks': ['Python AST', 'Python compile (no execution/import)', 'TOML parse', 'JSON parse/basic fields'],
          'runtime_tests_executed': False, 'gpu_tests_executed': False,
          'limitations': 'No dependency import, model load, numerical, training, performance, or CUDA validation.'}
(root/'docs/static-check.json').write_text(json.dumps(report, indent=2))
print(f"Static checks passed: {len(files)} Python files, project {metadata['project']['name']}")
