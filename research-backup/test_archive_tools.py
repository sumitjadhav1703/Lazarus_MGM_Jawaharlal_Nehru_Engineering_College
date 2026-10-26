import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import archive_tools as a


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'source'
        self.root.mkdir()
        (self.root / 'file').write_bytes(bytes(range(256)) * 20)
        (self.root / 'file').chmod(0o755)

    def test_roundtrip_multipart_hardlink(self):
        os.link(self.root / 'file', self.root / 'linked')
        out = self.base / 'backup'
        r = a.build(self.root, ['linked', 'file'], out, 100)
        self.assertGreater(len(r['parts']), 1)
        self.assertEqual(r['files'][1]['hardlink'], 'file')
        a.verify(out)
        destination = self.base / 'restore'
        a.verify(out, destination)
        self.assertEqual((destination / 'file').read_bytes(), (self.root / 'file').read_bytes())
        self.assertEqual((destination / 'file').stat().st_ino, (destination / 'linked').stat().st_ino)
        with self.assertRaises(ValueError):
            a.verify(out, destination)
        with self.assertRaises(FileExistsError):
            a.build(self.root, ['file'], out)

    def test_invalid_selection_and_symlink(self):
        for name in ['../outside', '/absolute', 'a/../file']:
            with self.assertRaises(ValueError):
                a.build(self.root, [name], self.base / 'bad')
        (self.root / 'link').symlink_to('file')
        with self.assertRaises(ValueError):
            a.build(self.root, ['link'], self.base / 'bad')

    def test_corruption_refused_before_restore(self):
        out = self.base / 'backup'
        a.build(self.root, ['file'], out, 100)
        p = next(out.glob('archive*'))
        p.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            a.verify(out, self.base / 'restore')
        self.assertFalse((self.base / 'restore').exists())

    def test_changed_source_leaves_incomplete(self):
        original = a.HashReader.read

        def mutate(reader, size=-1):
            data = original(reader, size)
            with (self.root / 'file').open('ab') as f:
                f.write(b'x')
            return data

        out = self.base / 'backup'
        with patch.object(a.HashReader, 'read', mutate), self.assertRaises(ValueError):
            a.build(self.root, ['file'], out)
        self.assertTrue((out / 'incomplete.json').exists())
        self.assertFalse((out / 'manifest.json').exists())
        with self.assertRaises(ValueError):
            a.verify(out)

    def test_malicious_tar_rejected_even_with_matching_part_hash(self):
        out = self.base / 'backup'
        a.build(self.root, ['file'], out)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz') as tar:
            info = tarfile.TarInfo('../escape')
            info.size = 1
            tar.addfile(info, io.BytesIO(b'x'))
        data = buf.getvalue()
        part = out / 'archive.tar.gz.part00000'
        part.write_bytes(data)
        manifest = json.loads((out / 'manifest.json').read_text())
        manifest['archive_bytes'] = len(data)
        manifest['archive_sha256'] = hashlib.sha256(data).hexdigest()
        manifest['parts'] = [dict(name=part.name, bytes=len(data), sha256=manifest['archive_sha256'])]
        (out / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            a.verify(out)
        self.assertFalse((self.base / 'escape').exists())


if __name__ == '__main__':
    unittest.main()
