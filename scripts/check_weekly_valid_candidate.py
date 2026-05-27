from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether a weekly leaderboard already has a valid candidate.")
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--rule", choices=("positive_calmar", "sharpe_robust"), default="positive_calmar")
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()

    rows = _read_csv(Path(args.leaderboard))
    valid = [row for row in rows if _is_valid(row, args.rule)]
    best = _best(valid, args.rule)
    summary = {
        "rule": args.rule,
        "rows": len(rows),
        "valid_count": len(valid),
        "found": len(valid) >= args.min_count,
        "best_candidate": best,
        "locked_opened": any(_to_bool(row.get("locked_opened")) for row in rows),
    }
    output_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"found={str(summary['found']).lower()}\n")
            handle.write(f"valid_count={summary['valid_count']}\n")
            handle.write(f"best_candidate_id={(best or {}).get('candidate_id', '')}\n")
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _is_valid(row: dict[str, Any], rule: str) -> bool:
    if _to_bool(row.get("locked_opened")):
        return False
    if rule == "sharpe_robust":
        return _to_bool(row.get("verified_sharpe_robust"))
    return _finite_float(row.get("train_calmar")) > 0.0 and _finite_float(row.get("validation_calmar")) > 0.0


def _best(rows: list[dict[str, Any]], rule: str) -> dict[str, Any] | None:
    if not rows:
        return None
    if rule == "sharpe_robust":
        key = lambda row: (_finite_float(row.get("validation_sharpe")), _finite_float(row.get("train_sharpe")))
    else:
        key = lambda row: (_finite_float(row.get("validation_calmar")), _finite_float(row.get("train_calmar")))
    row = max(rows, key=key)
    return {
        "candidate_id": row.get("candidate_id", ""),
        "method": row.get("method", ""),
        "train_calmar": _finite_float(row.get("train_calmar")),
        "validation_calmar": _finite_float(row.get("validation_calmar")),
        "train_sharpe": _finite_float(row.get("train_sharpe")),
        "validation_sharpe": _finite_float(row.get("validation_sharpe")),
        "train_cagr": _finite_float(row.get("train_cagr")),
        "validation_cagr": _finite_float(row.get("validation_cagr")),
    }


def _finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return out if math.isfinite(out) else float("-inf")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    raise SystemExit(main())
