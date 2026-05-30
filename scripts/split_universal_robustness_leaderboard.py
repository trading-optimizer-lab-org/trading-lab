from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def split_leaderboard(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    chunks: int = 64,
    file_prefix: str = "chunk",
    dedupe_key: str | None = None,
) -> dict[str, int | str]:
    """Split a leaderboard into deterministic chunks for GitHub matrix jobs."""

    if chunks < 1:
        raise ValueError("--chunks must be >= 1")
    source = pd.read_csv(input_path)
    input_rows = int(len(source))
    if dedupe_key:
        if dedupe_key not in source.columns:
            raise ValueError(f"--dedupe-key column not found: {dedupe_key}")
        source = source.drop_duplicates(subset=[dedupe_key], keep="first").reset_index(drop=True)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    size = max(1, (len(source) + int(chunks) - 1) // int(chunks))
    n_trials = int(len(source))
    manifest = []
    for index in range(int(chunks)):
        chunk = source.iloc[index * size : (index + 1) * size].copy()
        path = output / f"{file_prefix}_{index:02d}.csv"
        chunk.to_csv(path, index=False)
        manifest.append({"chunk": index, "path": str(path), "rows": int(len(chunk)), "n_trials": n_trials})
    summary = {
        "input_rows": input_rows,
        "rows": n_trials,
        "duplicates_removed": int(input_rows - n_trials),
        "chunks": int(chunks),
        "n_trials": n_trials,
        "output_dir": str(output),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Split verified leaderboard rows for universal robustness chunks.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunks", type=int, default=64)
    parser.add_argument("--file-prefix", default="chunk")
    parser.add_argument("--dedupe-key", default="", help="Optional column used to keep one row per candidate.")
    args = parser.parse_args()

    summary = split_leaderboard(
        args.input,
        args.output_dir,
        chunks=int(args.chunks),
        file_prefix=str(args.file_prefix),
        dedupe_key=str(args.dedupe_key).strip() or None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
