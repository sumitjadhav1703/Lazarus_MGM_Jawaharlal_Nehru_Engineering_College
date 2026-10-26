"""Official SHDB-AF files; full cohort is approximately 8.3 GB."""
import argparse
from pathlib import Path

from cardiotwin.data import download_shdb

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--records', nargs='+', help='IDs e.g. 001 002; omission means full cohort')
parser.add_argument('--metadata-only', action='store_true', help='All headers/annotations, without large ECGs')
parser.add_argument('--source-url', help='Optional official mirror, e.g. public PhysioNet S3 HTTPS endpoint')
args = parser.parse_args()
print(download_shdb(args.output, args.records, args.metadata_only, args.source_url))
