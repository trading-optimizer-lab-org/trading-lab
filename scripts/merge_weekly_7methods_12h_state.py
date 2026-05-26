from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.weekly_7methods_stateful import merge_state_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge weekly 7-method state files for the next wave.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-top", type=int, default=500)
    parser.add_argument("--expected-files-per-method", type=int, default=0)
    parser.add_argument("--allow-missing-files-per-method", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob, recursive=True))
    summary = merge_state_files(
        paths,
        args.output_dir,
        state_top=args.state_top,
        expected_files_per_method=args.expected_files_per_method,
        allow_missing_files_per_method=args.allow_missing_files_per_method,
    )
    print(json.dumps({"input_files": len(paths), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
