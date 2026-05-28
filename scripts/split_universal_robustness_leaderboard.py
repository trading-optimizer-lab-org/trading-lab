from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Split verified leaderboard rows for universal robustness chunks.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunks", type=int, default=64)
    parser.add_argument("--file-prefix", default="chunk")
    args = parser.parse_args()

    if args.chunks < 1:
        raise ValueError("--chunks must be >= 1")
    source = pd.read_csv(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    size = max(1, (len(source) + int(args.chunks) - 1) // int(args.chunks))
    manifest = []
    for index in range(int(args.chunks)):
        chunk = source.iloc[index * size : (index + 1) * size].copy()
        path = output / f"{args.file_prefix}_{index:02d}.csv"
        chunk.to_csv(path, index=False)
        manifest.append({"chunk": index, "path": str(path), "rows": int(len(chunk))})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"rows": int(len(source)), "chunks": int(args.chunks), "output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
