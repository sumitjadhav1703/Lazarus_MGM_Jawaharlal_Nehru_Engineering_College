"""Create the single private review capsule. Does not call any provider or run a worker."""
import ast
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = str(HERE.parent.relative_to(ROOT))
IDENTITY = 'sumit1703/cardiotwin-ecgfounder-native-cost-20261026-a'
DATASET = 'sumit1703/cardiotwin-iridia-development-gpu-inputs/1'
CALENDAR = BASE + '/forecast-calendar-scope-v3.json'
OLD = b'DEADLINE = datetime(2026, 10, 22, 12, 30, tzinfo=timezone.utc)'
NEW = b'DEADLINE = datetime(2026, 10, 27, 12, 30, tzinfo=timezone.utc)'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def source_payload(root=ROOT):
    root = Path(root)
    names = {'pyproject.toml', 'uv.lock', 'configs/default.json', 'configs/precursor-cycle.json',
             'scripts/kaggle_ecgfounder_engineering.py', 'scripts/kaggle_cycle.py',
             'scripts/kaggle_ecgfounder_probe.py', 'scripts/run_ecgfounder_session.py',
             'scripts/kaggle_ecgfounder_session.py', CALENDAR,
             BASE + '/forecast-reconsideration-science-v2.json',
             BASE + '/forecast-scope-envelope-clarification-v2a.md',
             BASE + '/forecast-table-scoring-scope-v3a.md',
             BASE + '/native-metadata-counts.json'}
    names.update(str(path.relative_to(root)) for path in (root / BASE / 'native-gap-next').glob('*.py')
                 if not path.name.startswith('test_'))
    names.update(str(path.relative_to(root)) for path in (root / BASE / 'native-gap-next').glob('*.json'))
    names.update(str(path.relative_to(root)) for path in HERE.glob('*.py'))
    names.add(BASE + '/launch-next/input-pins.json')
    pins = json.loads((root / BASE / 'launch-next/input-pins.json').read_text())
    names.update(name for name in pins if not name.startswith('data/'))
    names.add('src/cardiotwin/__init__.py')
    pending = list(names)
    while pending:
        name = pending.pop()
        if not name.endswith('.py'):
            continue
        for node in ast.walk(ast.parse((root / name).read_text())):
            module = None
            if isinstance(node, ast.ImportFrom):
                if node.level and name.startswith('src/cardiotwin/') and node.module:
                    module = 'cardiotwin.' + node.module
                elif node.module and node.module.startswith('cardiotwin.'):
                    module = node.module
                elif node.module == 'cardiotwin':
                    for alias in node.names:
                        dependency = 'src/cardiotwin/' + alias.name + '.py'
                        if (root / dependency).is_file() and dependency not in names:
                            names.add(dependency)
                            pending.append(dependency)
            if module:
                dependency = 'src/' + module.replace('.', '/') + '.py'
                if (root / dependency).is_file() and dependency not in names:
                    names.add(dependency)
                    pending.append(dependency)
    # Dynamic local script loaders are not discoverable through Python imports.
    for name in ('ecgfounder_runner', 'ecgfounder_probe'):
        names.add('src/cardiotwin/' + name + '.py')
    files, original, total = {}, {}, 0
    for name in sorted(names):
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or relative.parts[0] in ('data', '.git', '.venv')
                or (root / name).is_symlink()):
            raise ValueError('Unsafe source member: ' + name)
        data = (root / name).read_bytes()
        if len(data) > 5_000_000:
            raise ValueError('Oversized source member')
        original[name] = digest(data)
        if name in ('scripts/kaggle_ecgfounder_engineering.py', 'src/cardiotwin/ecgfounder_runner.py'):
            if data.count(OLD) != 1:
                raise ValueError('Expected original calendar differs')
            data = data.replace(OLD, NEW)
        if name == 'scripts/kaggle_ecgfounder_engineering.py':
            before = b'ENGINEERING_SECONDS = 41400, 300, 3600'
            if data.count(before) != 1:
                raise ValueError('Expected original engineering cap differs')
            data = data.replace(before, b'ENGINEERING_SECONDS = 41400, 300, 1800')
        files[name] = base64.b64encode(data).decode()
        total += len(data)
    if total > 10_000_000:
        raise ValueError('Source capsule exceeds10MB')
    manifest = {name: digest(base64.b64decode(data)) for name, data in files.items()}
    calendar = json.loads(base64.b64decode(files[CALENDAR]))
    if calendar['stop_utc'] != '2026-10-27T12:30:00Z' or calendar['measurement_maximum'] != 1:
        raise ValueError('Prospective single measurement calendar differs')
    return dict(identity_name=IDENTITY, dataset_sources=[DATASET], files=files,
                files_sha256=manifest, original_source_sha256=original,
                source_identity=digest(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()),
                source_bytes=total)


def prepare():
    raise RuntimeError('Source-only continuation: v3 closed; no executable package or dispatch released')


if __name__ == '__main__':
    result = prepare()
    print(json.dumps({key: result[key] for key in ('source_identity', 'source_bytes')}))
