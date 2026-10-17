"""Build a data dir that serves exactly ONE IRIDIA-AF record (header + its row, symlinked ECG)."""

import argparse
import csv
import json
import sys
from pathlib import Path


def prepare(source, record, output, split_file=None):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a fresh path")
    with (source / "metadata.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row["record_id"] == record]
        header = reader.fieldnames
    if len(rows) != 1:
        raise ValueError(f"expected exactly one metadata row for {record}, found {len(rows)}")
    if not (source / "records" / record).is_dir():
        raise FileNotFoundError(f"{source / 'records' / record} not found")
    if split_file:
        split = json.loads(Path(split_file).read_text())
        patient = f"iridia-af:{rows[0]['patient_id']}"
        group = split.get("group_ids", {}).get(patient, patient)
        if group in set(split["test_groups"]):
            raise ValueError(f"{record} belongs to a held-out test group; refusing to replay it")
    (output / "records").mkdir(parents=True)
    with (output / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerow(rows[0])
    (output / "records" / record).symlink_to((source / "records" / record).resolve(), target_is_directory=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="IRIDIA-AF data dir (metadata.csv + records/)")
    parser.add_argument("--record", required=True, help="record id, e.g. record_000")
    parser.add_argument("--output", required=True, help="fresh output directory")
    parser.add_argument("--split-file", help="frozen split JSON; refuse records in its test groups")
    args = parser.parse_args(argv)
    try:
        output = prepare(args.source, args.record, args.output, args.split_file)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Prepared {output}. Serve it with:")
    print(f"uv run cardiotwin serve --research-model --model <frozen model dir> --data-dir {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
