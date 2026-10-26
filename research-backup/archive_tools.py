"""Stdlib, streaming split-tar preservation. Python 3.12+ required for restore."""
import argparse
import gzip
import hashlib
import json
import os
import stat
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def safe_name(name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError(f'Invalid archive path: {name!r}')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p) != name or name == '.':
        raise ValueError(f'Unsafe archive path: {name!r}')
    return name


def source(root, name):
    path = root
    for component in PurePosixPath(safe_name(name)).parts:
        path /= component
        if path.is_symlink():
            raise ValueError(f'Symlink excluded: {name}')
    return path


def fingerprint(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)


def write_json(path, obj):
    temporary = path.with_suffix('.tmp')
    with temporary.open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


class Parts:
    def __init__(self, directory, limit):
        self.directory, self.limit = directory, limit
        self.parts, self.file, self.size = [], None, 0
        self.total, self.digest = 0, hashlib.sha256()

    def write(self, data):
        original = len(data)
        self.digest.update(data)
        while data:
            if self.file is None:
                name = f'archive.tar.gz.part{len(self.parts):05d}'
                self.file = (self.directory / name).open('xb')
                self.parts.append({'name': name, 'bytes': 0})
                self.part_digest = hashlib.sha256()
                self.size = 0
            chunk = data[:self.limit - self.size]
            self.file.write(chunk)
            self.part_digest.update(chunk)
            self.size += len(chunk)
            self.total += len(chunk)
            data = data[len(chunk):]
            if self.size == self.limit:
                self.finish_part()
        return original

    def flush(self):
        if self.file:
            self.file.flush()

    def finish_part(self):
        if self.file:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()
            self.parts[-1].update(bytes=self.size, sha256=self.part_digest.hexdigest())
            self.file = None


class HashReader:
    def __init__(self, f, tick):
        self.f, self.tick, self.digest = f, tick, hashlib.sha256()

    def read(self, size=-1):
        data = self.f.read(size)
        self.digest.update(data)
        self.tick()
        return data


def build(root, selection, output, part_bytes=1_000_000_000):
    root, output = Path(root).resolve(), Path(output)
    names = sorted(safe_name(n) for n in selection)
    if len(names) != len(set(names)) or not names or part_bytes <= 0:
        raise ValueError('Selection must be nonempty and unique; part size must be positive')
    snapshots = {}
    for name in names:
        s = source(root, name).stat()
        if not stat.S_ISREG(s.st_mode):
            raise ValueError(f'Not a regular file: {name}')
        snapshots[name] = s
    output.mkdir(parents=False, exist_ok=False)
    if output.resolve().is_relative_to(root) and any(
        source(root, n).is_relative_to(output.resolve()) for n in names
    ):
        raise ValueError('Output overlaps selected source')
    started, last = time.monotonic(), [0.0]
    writer = Parts(output, part_bytes)
    current, completed = [''], [0]

    def tick(force=False):
        now = time.monotonic()
        if force or now - last[0] >= 5:
            print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), phase='archive',
                  current_file=current[0], completed=completed[0], total=len(names),
                  elapsed_seconds=round(now-started, 3), output_bytes=writer.total)), flush=True)
            last[0] = now

    write_json(output / 'incomplete.json', {'status': 'incomplete', 'selected_file_count': len(names)})
    entries, links = [], {}
    try:
        with gzip.GzipFile(fileobj=writer, mode='wb', filename='', mtime=0, compresslevel=1) as zipped:
            with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                for name in names:
                    current[0] = name
                    s = snapshots[name]
                    info = tarfile.TarInfo(name)
                    info.mode, info.mtime = stat.S_IMODE(s.st_mode), s.st_mtime
                    key = (s.st_dev, s.st_ino)
                    entry = dict(path=name, bytes=s.st_size, mode=info.mode)
                    if key in links:
                        prior = links[key]
                        info.type, info.linkname = tarfile.LNKTYPE, prior['path']
                        entry.update(sha256=prior['sha256'], hardlink=prior['path'])
                        archive.addfile(info)
                    else:
                        info.size = s.st_size
                        fd = os.open(source(root, name), os.O_RDONLY | os.O_NOFOLLOW)
                        with os.fdopen(fd, 'rb') as f:
                            if fingerprint(os.fstat(f.fileno())) != fingerprint(s):
                                raise ValueError(f'Source changed: {name}')
                            reader = HashReader(f, tick)
                            archive.addfile(info, reader)
                            if fingerprint(os.fstat(f.fileno())) != fingerprint(s):
                                raise ValueError(f'Source changed: {name}')
                        entry['sha256'] = reader.digest.hexdigest()
                        links[key] = entry
                    entries.append(entry)
                    completed[0] += 1
                    tick(completed[0] % 50 == 0)
        writer.finish_part()
        for name, s in snapshots.items():
            if fingerprint(source(root, name).stat()) != fingerprint(s):
                raise ValueError(f'Source changed: {name}')
        receipt = dict(status='complete', format=1, selected_file_count=len(entries), files=entries,
                       parts=writer.parts, archive_bytes=writer.total, archive_sha256=writer.digest.hexdigest())
        write_json(output / 'manifest.json', receipt)
        (output / 'incomplete.json').unlink()
        tick(True)
        return receipt
    except BaseException:
        writer.finish_part()
        raise


class Joined:
    def __init__(self, paths):
        self.paths, self.f = iter(paths), None

    def read(self, size=-1):
        chunks, remaining = [], size
        while remaining:
            if self.f is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.f = path.open('rb')
            data = self.f.read(remaining)
            if not data:
                self.f.close()
                self.f = None
                continue
            chunks.append(data)
            if size >= 0:
                remaining -= len(data)
        return b''.join(chunks)

    def close(self):
        if self.f:
            self.f.close()


def verify(directory, restore=None):
    directory = Path(directory)
    if (directory / 'incomplete.json').exists():
        raise ValueError('Incomplete archive')
    receipt = json.loads((directory / 'manifest.json').read_text())
    if receipt['status'] != 'complete' or receipt['format'] != 1:
        raise ValueError('Unsupported/incomplete manifest')
    expected = {safe_name(e['path']): e for e in receipt['files']}
    if len(expected) != len(receipt['files']) or len(expected) != receipt['selected_file_count']:
        raise ValueError('Duplicate/inconsistent manifest')
    total, whole, paths = 0, hashlib.sha256(), []
    started, last = time.monotonic(), [0.0]

    def progress(phase, current, completed, force=False):
        now = time.monotonic()
        if force or now - last[0] >= 5:
            print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), phase=phase,
                  current_file=current, completed=completed, total=len(expected),
                  elapsed_seconds=round(now-started, 3), output_bytes=total)), flush=True)
            last[0] = now
    for i, part in enumerate(receipt['parts']):
        if part['name'] != f'archive.tar.gz.part{i:05d}':
            raise ValueError('Invalid part sequence')
        path, digest, size = directory / part['name'], hashlib.sha256(), 0
        if path.is_symlink():
            raise ValueError('Symlink part')
        with path.open('rb') as f:
            while data := f.read(1024 * 1024):
                size += len(data)
                digest.update(data)
                whole.update(data)
                progress('verify_parts', part['name'], 0)
        if size != part['bytes'] or digest.hexdigest() != part['sha256']:
            raise ValueError(f'Corrupt part: {part["name"]}')
        total += size
        paths.append(path)
    if total != receipt['archive_bytes'] or whole.hexdigest() != receipt['archive_sha256']:
        raise ValueError('Corrupt complete archive')
    destination = Path(restore) if restore is not None else None
    if destination is not None:
        if destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
            raise ValueError('Restore destination must be absent or empty')
        destination.mkdir(parents=True, exist_ok=True)
    joined, seen = Joined(paths), {}
    try:
        with gzip.GzipFile(fileobj=joined, mode='rb') as zipped:
            with tarfile.open(fileobj=zipped, mode='r|') as archive:
                for member in archive:
                    name = safe_name(member.name)
                    if name not in expected or name in seen or not (member.isfile() or member.islnk()):
                        raise ValueError(f'Unexpected/duplicate/unsafe member: {name}')
                    entry = expected[name]
                    if member.mode != entry['mode']:
                        raise ValueError(f'Mode mismatch: {name}')
                    if member.islnk():
                        target = safe_name(member.linkname)
                        if target not in seen or expected[target].get('hardlink') or entry.get('hardlink') != target:
                            raise ValueError(f'Unsafe hardlink: {name}')
                        digest, size = seen[target]
                    else:
                        if entry.get('hardlink'):
                            raise ValueError(f'Hardlink mismatch: {name}')
                        digestor, size = hashlib.sha256(), 0
                        f = archive.extractfile(member)
                        while data := f.read(1024 * 1024):
                            digestor.update(data)
                            size += len(data)
                            progress('verify_contents', name, len(seen))
                        digest = digestor.hexdigest()
                    if digest != entry['sha256'] or size != entry['bytes']:
                        raise ValueError(f'Content mismatch: {name}')
                    seen[name] = digest, size
                    progress('verify_contents', name, len(seen), len(seen) % 50 == 0)
            while zipped.read(1024 * 1024):
                pass  # Consume the gzip trailer and validate CRC, including after tar end markers.
    finally:
        joined.close()
    if set(seen) != set(expected):
        raise ValueError('Missing archive members')
    if destination is not None:
        joined = Joined(paths)
        try:
            with gzip.GzipFile(fileobj=joined, mode='rb') as zipped:
                with tarfile.open(fileobj=zipped, mode='r|') as archive:
                    for member in archive:
                        archive.extract(member, destination, filter='data')
        finally:
            joined.close()
        actual = set()
        for p in destination.rglob('*'):
            if p.is_symlink():
                raise ValueError('Unexpected restored symlink')
            if p.is_file():
                name = p.relative_to(destination).as_posix()
                actual.add(name)
                e = expected[name]
                h = hashlib.sha256()
                with p.open('rb') as f:
                    while data := f.read(1024 * 1024):
                        h.update(data)
                if p.stat().st_size != e['bytes'] or h.hexdigest() != e['sha256'] or (stat.S_IMODE(p.stat().st_mode) & 0o111) != (e['mode'] & 0o111):
                    raise ValueError(f'Restored file mismatch: {name}')
        if actual != set(expected):
            raise ValueError('Restored inventory mismatch')
    progress('verified', '', len(seen), True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--root', required=True)
    b.add_argument('--selection', required=True)
    b.add_argument('--output', required=True)
    b.add_argument('--part-bytes', type=int, default=1_000_000_000)
    v = sub.add_parser('verify')
    v.add_argument('--archive-dir', required=True)
    v.add_argument('--restore')
    args = parser.parse_args()
    if args.command == 'build':
        build(args.root, json.loads(Path(args.selection).read_text()), args.output, args.part_bytes)
    else:
        verify(args.archive_dir, args.restore)
        print('Verified all compressed parts and selected file contents.', flush=True)
