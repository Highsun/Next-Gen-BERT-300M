#!/usr/bin/env python3
"""CPU-only final experiment archive, including trusted weights and cached base.

No model import, training, inference, dependency installation, or source deletion.
Archive verification reads bytes without extracting or unpickling checkpoints.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import posixpath
import shutil
import subprocess
import tarfile


MODEL_CACHE = 'models--jhu-clsp--mmBERT-base'
ARMS = ('B0', 'L3', 'G2')
ROOT_FILES = ('README.md', 'AGENTS.md', 'LICENSE', 'NOTICE', 'pyproject.toml', '.gitignore')
PROJECT_DIRS = ('src', 'scripts', 'tests', 'configs', 'docs', 'data', 'outputs', 'runs', 'logs', 'raw')
SKIP_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache', '.ruff_cache', '.ipynb_checkpoints', '.locks'}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def ignored(path):
    name = path.name
    return (any(part in SKIP_DIRS for part in path.parts)
            or name == '.DS_Store' or name.startswith('.env')
            or name in {'token', 'stored_tokens', 'id_rsa', 'id_ed25519'}
            or name.endswith(('.pyc', '.pyo', '.pem', '.key', '.incomplete')))


def tree_files(root):
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not ignored(Path(directory) / d))
        for name in sorted(files):
            path = Path(directory) / name
            if not ignored(path):
                yield path
        # Refuse directory symlinks instead of silently omitting their contents.
        if any((Path(directory) / d).is_symlink() for d in dirs):
            raise ValueError(f'Directory symlink needs explicit review: {directory}')


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {'returncode': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'unavailable': type(exc).__name__}


def add_json(archive, name, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    item = tarfile.TarInfo(name)
    item.size = len(data)
    item.mode = 0o644
    archive.addfile(item, io.BytesIO(data))
    return hashlib.sha256(data).hexdigest()


def validate_evidence(project):
    """Cross-check exported identities and saved files without model execution."""
    stage = project / 'outputs/stage1-s17'
    diag = project / 'outputs/stage1-diagnostics-s1701'
    load = lambda path: json.loads(path.read_text())
    frozen, summary = load(stage / 'stage-manifest.json'), load(stage / 'summary.json')
    corpus = project / 'data/wiki-phase1'
    source = load(corpus / 'manifest.json')
    for split in ('train', 'dev'):
        if sha256(corpus / f'{split}.jsonl') != source['files'][split] or source['files'] != frozen['data_sha256']:
            raise ValueError('Frozen corpus hashes changed')
    for lang, value in frozen['dev_file_sha256'].items():
        if sha256(stage / f'dev.{lang}.jsonl') != value:
            raise ValueError('Stage1 held-out data changed')
    dm = load(diag / 'manifest.json')
    if sha256(diag / 'cases.jsonl') != dm['cases_sha256'] or load(diag / 'summary.json')['manifest'] != dm:
        raise ValueError('Diagnostic inputs/summary identity mismatch')
    for name, value in dm['implementation_sha256'].items():
        if sha256(project / name) != value:
            raise ValueError(f'Implementation changed since diagnostics: {name}')
    for arm in ('P0',) + ARMS:
        metrics = load(diag / f'{arm}-metrics.json')
        if (metrics['identity']['manifest_sha256'] != sha256(diag / 'manifest.json')
                or metrics['identity']['arm'] != arm
                or sha256(diag / f'{arm}-predictions.jsonl') != metrics['predictions_sha256']):
            raise ValueError(f'Diagnostic predictions/identity changed: {arm}')
        if arm == 'P0':
            continue
        run = project / f'runs/stage1-s17/{arm}'
        records = [json.loads(line) for line in (run / 'train.jsonl').read_text().splitlines() if line.strip()]
        if ([r['step'] for r in records] != list(range(1, 1001))
                or not all(math.isfinite(r['loss']) for r in records)
                or records[-1] != summary['results'][arm]['training']
                or any(r['data_sha256'] != frozen['data_sha256']['train'] for r in records)):
            raise ValueError(f'Training log/summary/data mismatch: {arm}')
        if sha256(run / 'final/checkpoint.pt') != metrics['identity']['checkpoint_sha256']:
            raise ValueError(f'Final checkpoint differs from the diagnostic checkpoint: {arm}')
        final_config = load(run / 'final/config.json')
        if final_config != load(run / 'provenance.json')['config']:
            raise ValueError(f'Final config/provenance mismatch: {arm}')
        if sha256(stage / f'{arm}.json') != dm['stage_config_sha256'][arm]:
            raise ValueError(f'Stage config changed: {arm}')
        for lang in ('zh', 'en'):
            result = load(stage / f'{arm}-dev-{lang}.json')
            if result != summary['results'][arm]['dev'][lang]:
                raise ValueError(f'Stage dev metrics mismatch: {arm}/{lang}')
    return {'frozen_corpus': True, 'stage_train_logs': True, 'stage_configs_and_metrics': True,
            'diagnostic_cases_and_predictions': True, 'final_checkpoint_identities': True,
            'diagnostic_implementation': True, 'no_model_execution': True}


def verify(path):
    """Check every payload member against the embedded manifest; never extract."""
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError('Archive contains duplicate paths')
        for item in members:
            if item.name.startswith('/') or '..' in Path(item.name).parts:
                raise ValueError('Unsafe archive path')
        manifest_member = archive.extractfile('ARCHIVE-MANIFEST.json')
        if manifest_member is None:
            raise ValueError('Missing embedded manifest')
        manifest = json.load(manifest_member)
        expected = manifest['files']
        if set(names) != set(expected) | {'ARCHIVE-MANIFEST.json'}:
            raise ValueError('Archive member set differs from manifest')
        for i, item in enumerate(members):
            if item.name == 'ARCHIVE-MANIFEST.json':
                continue
            info = expected[item.name]
            if item.issym():
                target = posixpath.normpath(posixpath.join(posixpath.dirname(item.name), item.linkname))
                if (info['type'] != 'symlink' or info['target'] != item.linkname
                        or target.startswith('../') or target not in expected
                        or expected[target]['type'] != 'file'):
                    raise ValueError(f'Invalid/missing symlink target: {item.name}')
            elif item.isfile() and info['type'] == 'file':
                value = hashlib.sha256()
                handle = archive.extractfile(item)
                if handle is None:
                    raise ValueError('Missing regular file payload')
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    value.update(chunk)
                if item.size != info['size'] or value.hexdigest() != info['sha256']:
                    raise ValueError(f'Checksum mismatch: {item.name}')
            else:
                raise ValueError(f'Unsupported archive member: {item.name}')
            if i and i % 250 == 0:
                print(f'Verified {i} members', flush=True)
    print(f'ARCHIVE VERIFIED: {len(expected)} payload members; all SHA-256 checks passed.', flush=True)
    return manifest


def build(project, output):
    project, output = project.resolve(), output.resolve()
    if output == project or project in output.parents:
        raise ValueError('Backup directory must be outside the project')
    if output.exists():
        raise FileExistsError('Choose a new --output directory; do not overwrite a backup')
    required = ['outputs/stage1-s17/summary.json', 'outputs/stage1-s17/stage-manifest.json',
                'data/wiki-phase1/manifest.json', 'data/wiki-phase1/train.jsonl', 'data/wiki-phase1/dev.jsonl',
                'outputs/stage1-diagnostics-s1701/manifest.json',
                'outputs/stage1-diagnostics-s1701/summary.json',
                'outputs/stage1-diagnostics-s1701/cases.jsonl',
                'logs/stage1-s17.log', 'logs/stage1-diagnostics-s1701.log']
    for arm in ARMS:
        required.extend(f'runs/stage1-s17/{arm}/{name}' for name in
                        ('train.jsonl', 'provenance.json', 'final/config.json', 'final/checkpoint.pt'))
        required.append(f'outputs/stage1-s17/{arm}.json')
        for lang in ('zh', 'en'):
            required.extend((f'outputs/stage1-s17/{arm}-dev-{lang}.json',
                             f'outputs/stage1-s17/{arm}-dev-{lang}.json.predictions.jsonl',
                             f'outputs/stage1-s17/dev.{lang}.jsonl'))
    for arm in ('P0',) + ARMS:
        required.extend(f'outputs/stage1-diagnostics-s1701/{arm}-{suffix}' for suffix in
                        ('metrics.json', 'predictions.jsonl'))
    missing = sorted({name for name in required if not (project / name).is_file()})
    if missing:
        raise FileNotFoundError('Required experiment files missing:\n' + '\n'.join(missing))
    stage_summary = json.loads((project / 'outputs/stage1-s17/summary.json').read_text())
    for arm in ARMS:
        if stage_summary['results'][arm]['training']['step'] != 1000:
            raise ValueError('Stage1 is incomplete')
    diag = json.loads((project / 'outputs/stage1-diagnostics-s1701/manifest.json').read_text())
    for arm in ('P0',) + ARMS:
        metrics = json.loads((project / f'outputs/stage1-diagnostics-s1701/{arm}-metrics.json').read_text())
        if metrics['records'] != diag['case_count']:
            raise ValueError('Diagnostics are incomplete')
    cfg = json.loads((project / 'outputs/stage1-s17/B0.json').read_text())
    evidence_checks = validate_evidence(project)
    print('Evidence preflight passed, including final checkpoint SHA-256 identities.', flush=True)
    historical = ('outputs/pilot-B0.json', 'outputs/pilot-G2.json',
                  'outputs/B0-smoke-dev.json', 'outputs/G2-smoke-dev.json',
                  'data/wiki-starter/manifest.json', 'data/wiki-long-check/training-summary.json',
                  'data/wiki-long-check/B0-operators.txt', 'data/wiki-long-check/G2-operators.txt',
                  'data/wiki-long-check-v2/training-summary.json')
    historical_status = {name: (project / name).is_file() for name in historical}
    for name, exists in historical_status.items():
        if not exists:
            print(f'Historical artifact absent (no rerun requested): {name}', flush=True)
    hf_home = Path(os.environ.get('HF_HOME', '/root/autodl-tmp/hf-cache')).expanduser()
    cache_roots = [hf_home / 'hub', hf_home]
    if os.environ.get('HF_HUB_CACHE'):
        cache_roots.insert(0, Path(os.environ['HF_HUB_CACHE']).expanduser())
    cache = next((root / MODEL_CACHE for root in cache_roots if (root / MODEL_CACHE).is_dir()), None)
    if cache is None:
        raise FileNotFoundError('Base-model cache not found; export the actual HF_HOME or HF_HUB_CACHE')
    snapshot = cache / 'snapshots' / cfg['revision']
    if (not (snapshot / 'config.json').is_file()
            or not any(f.is_file() for pattern in ('*.safetensors', '*.bin') for f in snapshot.glob(pattern))):
        raise FileNotFoundError('Pinned base snapshot/config/weights are incomplete')
    entries = []
    for name in ROOT_FILES:
        path = project / name
        if path.is_file():
            entries.append((path, 'project/' + name, project))
    for name in PROJECT_DIRS:
        directory = project / name
        if directory.is_dir():
            entries.extend((path, 'project/' + str(path.relative_to(project)), project)
                           for path in tree_files(directory))
    entries.extend((path, 'hf-cache/hub/' + MODEL_CACHE + '/' + str(path.relative_to(cache)), cache)
                   for path in tree_files(cache))
    entries.sort(key=lambda item: item[1])
    total_bytes = sum(path.lstat().st_size for path, _, _ in entries if not path.is_symlink())
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < total_bytes * 1.1 + 1024 ** 3:
        raise OSError('Need at least 1.1 x source payload size + 1 GiB free on the backup disk')
    print(f'Payload: {len(entries)} members, {total_bytes / 1024**3:.2f} GiB before compression.', flush=True)
    print('CPU-only archiving; do not train, edit, or delete source files while this runs.', flush=True)
    output.mkdir()
    archive_path = output / 'nextbert-final-backup.tar.gz'
    temporary = output / 'nextbert-final-backup.tar.gz.tmp'
    original_states, manifest_files = {}, {}
    with tarfile.open(temporary, 'w:gz', compresslevel=3, dereference=False) as archive:
        for i, (path, name, boundary) in enumerate(entries, 1):
            before = path.lstat()
            original_states[path] = (before.st_size, before.st_mtime_ns, before.st_ino)
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                if boundary.resolve() not in resolved.parents:
                    raise ValueError(f'Symlink leaves archived tree: {path}')
                manifest_files[name] = {'type': 'symlink', 'target': os.readlink(path)}
                archive.add(path, arcname=name, recursive=False)
            elif path.is_file():
                manifest_files[name] = {'type': 'file', 'size': before.st_size, 'sha256': sha256(path)}
                item = archive.gettarinfo(path, arcname=name)
                # Store each regular file as bytes even if source files are hardlinked.
                # This keeps recovery independent of inode identity and member order.
                item.type, item.linkname, item.size = tarfile.REGTYPE, '', before.st_size
                with path.open('rb') as handle:
                    archive.addfile(item, handle)
            else:
                raise ValueError(f'Unsupported source file: {path}')
            if i % 100 == 0:
                print(f'Archived {i}/{len(entries)} members', flush=True)
        versions = sorted({(dist.metadata['Name'], dist.version) for dist in metadata.distributions()
                           if dist.metadata['Name']})
        environment = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                       'python': platform.python_version(), 'platform': platform.platform(),
                       'packages': [{'name': name, 'version': version} for name, version in versions],
                       'git_head': command(['git', '-C', str(project), 'rev-parse', 'HEAD']),
                       'git_status': command(['git', '-C', str(project), 'status', '--short']),
                       'gpu_readonly_query': command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                                     '--format=csv,noheader']),
                       'note': 'No environment variables, authentication tokens, remotes, or model imports collected.'}
        value = add_json(archive, 'ENVIRONMENT.json', environment)
        manifest_files['ENVIRONMENT.json'] = {'type': 'file',
            'size': len((json.dumps(environment, ensure_ascii=False, indent=2) + '\n').encode()), 'sha256': value}
        manifest = {'version': 1, 'created_at_utc': environment['created_at_utc'], 'files': manifest_files,
                    'model_name': cfg['model_name'], 'model_revision': cfg['revision'],
                    'required_experiment_files': sorted(set(required)),
                    'evidence_preflight': evidence_checks,
                    'historical_artifacts_present': historical_status,
                    'project_source': str(project), 'base_cache_source': str(cache),
                    'exclusions': ['virtual environment, Git history, auth files, temporary/incomplete files, Python caches'],
                    'limitations': ['Not a system/driver/container image; packages must be reinstalled for future execution.',
                                    'No new model evaluation or performance measurement is performed.',
                                    'Missing historical optional artifacts cannot be recovered by archiving.']}
        add_json(archive, 'ARCHIVE-MANIFEST.json', manifest)
    # Detect source mutation during the complete operation, not just individual reads.
    for path, original in original_states.items():
        now = path.lstat()
        if (now.st_size, now.st_mtime_ns, now.st_ino) != original:
            raise RuntimeError(f'Source changed during backup; do not release instance: {path}')
    verify(temporary)
    temporary.replace(archive_path)
    archive_hash = sha256(archive_path)
    (output / 'SHA256SUMS.txt').write_text(f'{archive_hash}  {archive_path.name}\n')
    (output / 'ARCHIVE-MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'BACKUP COMPLETE: {archive_path}', flush=True)
    print(f'Archive size: {archive_path.stat().st_size / 1024**3:.2f} GiB; SHA256: {archive_hash}', flush=True)
    print('Download the archive and SHA256SUMS.txt; verify the LOCAL copy before releasing the instance.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, default=Path('/root/autodl-tmp/nextbert-final-backup'))
    p.add_argument('--verify', type=Path, help='Verify an existing archive without extraction/model execution')
    a = p.parse_args()
    if a.verify:
        verify(a.verify)
    else:
        build(a.project, a.output)


if __name__ == '__main__':
    main()
