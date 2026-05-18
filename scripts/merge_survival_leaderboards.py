from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.results import merge_leaderboards, save_leaderboard


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge survival stage leaderboards.")
    parser.add_argument("--input-glob", default="outputs/survival/survival_stage_*.csv")
    parser.add_argument("--output", default="outputs/survival/survival_leaderboard.csv")
    parser.add_argument("--summary", default="outputs/survival/survival_summary.json")
    parser.add_argument("--candidates", default="outputs/survival/survival_candidates.jsonl")
    parser.add_argument("--rejected", default="outputs/survival/rejected_candidates.jsonl")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    leaderboard = merge_leaderboards(paths, score_column="survival_score")
    output_path = save_leaderboard(leaderboard, args.output)
    accepted = leaderboard[leaderboard.get("accepted", False) == True] if not leaderboard.empty else leaderboard
    rejected = leaderboard[leaderboard.get("accepted", False) == False] if not leaderboard.empty else leaderboard
    _write_jsonl(accepted, args.candidates)
    _write_jsonl(rejected, args.rejected)
    summary = {
        "input_files": paths,
        "rows": int(len(leaderboard)),
        "candidates_evaluated": int(len(leaderboard)),
        "accepted": int(len(accepted)),
        "rejection_counts": _value_counts(leaderboard, "rejection_reason"),
        "output": str(output_path),
        "best": {} if leaderboard.empty else leaderboard.iloc[0].to_dict(),
        "locked_opened": False,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _write_jsonl(frame, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")


def _value_counts(frame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).items()}


if __name__ == "__main__":
    raise SystemExit(main())
