"""Acquire official reviewed beat/noise benchmarks, with local reproducibility hashes."""

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def acquire(dataset, output):
    base = f"https://physionet.org/files/{dataset}/1.0.0/"
    output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(base + "RECORDS", timeout=30) as response:
        names = response.read().decode().split()
    (output / "RECORDS").write_text("\n".join(names) + "\n")

    def fetch(name):
        path = output / name
        if not path.exists():
            part = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(base + name, timeout=60) as response, part.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            part.replace(path)
        with path.open("rb") as stream:
            return name, hashlib.file_digest(stream, "sha256").hexdigest()

    files = [
        f"{name}.{suffix}"
        for name in names
        for suffix in ("hea", "dat", "atr")
        if dataset != "nstdb" or name not in {"bw", "em", "ma"} or suffix != "atr"
    ]
    with ThreadPoolExecutor(max_workers=6) as executor:
        hashes = dict(executor.map(fetch, files))
    report = {
        "dataset": dataset,
        "source": base,
        "records": names,
        "local_sha256": hashes,
        "checksum_note": "Local SHA256 ensures reproducibility; not independent published checksum verification.",
        "license": "https://physionet.org/content/" + dataset + "/view-license/1.0.0/",
    }
    (output / "acquisition.json").write_text(json.dumps(report, indent=2) + "\n")
    print(dataset, len(names), "records acquired", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["mitdb", "nstdb"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    acquire(args.dataset, args.output)
