from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.config import load_yaml  # noqa: E402
from trading_lab.monthly_risk import _json_safe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge weekly SPY Sharpe 10-method state files.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/weekly_spy_sharpe_10methods_9h_waves.yaml")
    parser.add_argument("--state-top", type=int, default=500)
    parser.add_argument("--expected-files-per-method", type=int, default=0)
    parser.add_argument("--allow-missing-files-per-method", type=int, default=0)
    parser.add_argument("--file-prefix", default="weekly_spy_sharpe_10methods_9h")
    args = parser.parse_args()

    raw_config = load_yaml(args.config)
    methods = [str(item) for item in raw_config.get("methods", [])]
    paths = sorted(glob.glob(args.input_glob, recursive=True))
    summary = merge_state_files(
        paths,
        args.output_dir,
        methods=methods,
        state_top=args.state_top,
        expected_files_per_method=args.expected_files_per_method,
        allow_missing_files_per_method=args.allow_missing_files_per_method,
        file_prefix=args.file_prefix,
    )
    print(json.dumps({"input_files": len(paths), **summary}, indent=2, sort_keys=True))
    return 0


def merge_state_files(
    paths: list[str],
    output_dir: str | Path,
    *,
    methods: list[str],
    state_top: int,
    expected_files_per_method: int,
    allow_missing_files_per_method: int,
    file_prefix: str,
) -> dict[str, Any]:
    output = Path(output_dir)
    state_output = output / "state"
    state_output.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    files_by_method: dict[str, int] = {method: 0 for method in methods}
    raw_states = 0
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        method = str(payload.get("method", ""))
        if method not in grouped:
            continue
        raw_states += 1
        files_by_method[method] += 1
        grouped[method].extend([item for item in payload.get("candidates", []) if isinstance(item, dict)])

    allowed_minimum = max(0, int(expected_files_per_method) - max(0, int(allow_missing_files_per_method)))
    missing = {
        method: count
        for method, count in files_by_method.items()
        if expected_files_per_method > 0 and count < allowed_minimum
    }
    if missing:
        raise ValueError(
            "missing weekly SPY Sharpe 10-method state files: "
            + ", ".join(f"{method}={count}/{expected_files_per_method}" for method, count in sorted(missing.items()))
        )

    methods_summary = []
    for method in methods:
        candidates = _dedupe_candidates(grouped.get(method, []))
        kept = sorted(candidates, key=lambda row: float(row.get("train_score", 0.0) or 0.0), reverse=True)[: max(1, state_top)]
        state = {
            "method": method,
            "candidate_count": len(kept),
            "candidates": kept,
            "source_state_count": raw_states,
            "validation_role": "report_only",
            "locked_opened": False,
        }
        (state_output / f"{method}.json").write_text(json.dumps(_json_safe(state), indent=2, sort_keys=True), encoding="utf-8")
        methods_summary.append({"method": method, "state_candidates": len(kept), "state_files": files_by_method.get(method, 0)})

    summary = {
        "state_files": raw_states,
        "state_files_by_method": files_by_method,
        "expected_files_per_method": int(expected_files_per_method),
        "allow_missing_files_per_method": int(allow_missing_files_per_method),
        "methods": methods_summary,
        "validation_role": "report_only",
        "locked_opened": False,
    }
    (output / f"{file_prefix}_state_summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedup: dict[str, dict[str, Any]] = {}
    for row in candidates:
        key = str(row.get("candidate_id") or json.dumps(row, sort_keys=True))
        current = dedup.get(key)
        if current is None or float(row.get("train_score", 0.0) or 0.0) > float(current.get("train_score", 0.0) or 0.0):
            dedup[key] = row
    return list(dedup.values())


if __name__ == "__main__":
    raise SystemExit(main())
