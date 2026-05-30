from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FILES = (
    "universal_robustness_results.csv",
    "universal_robustness_pass.csv",
    "universal_robustness_methods.csv",
    "universal_robustness_year_by_year.csv",
    "universal_robustness_fail_reasons.csv",
    "universal_robustness_duplicates.csv",
    "universal_robustness_bootstrap.csv",
    "universal_robustness_data_quality.csv",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge universal robustness chunk artifacts.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-chunks", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    merged: dict[str, pd.DataFrame] = {}
    for name in FILES:
        frames = []
        for path in sorted(input_root.rglob(name)):
            if path.stat().st_size > 0:
                frames.append(pd.read_csv(path))
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        merged[name] = frame
        frame.to_csv(output / name, index=False)

    results = merged["universal_robustness_results.csv"]
    pass_results = results[results["robust_pass"].fillna(False).astype(bool)].copy() if "robust_pass" in results else pd.DataFrame()
    pass_results.to_csv(output / "universal_robustness_pass.csv", index=False)
    methods = _method_summary(results)
    methods.to_csv(output / "universal_robustness_methods.csv", index=False)

    chunk_dirs = {
        path.parent
        for path in input_root.rglob("universal_robustness_results.csv")
        if path.stat().st_size > 0
    }
    summary = {
        "input_candidates": int(len(results)),
        "chunks_found": int(len(chunk_dirs)),
        "expected_chunks": int(args.expected_chunks),
        "partial": bool(len(chunk_dirs) < int(args.expected_chunks)),
        "robust_pass": _sum_bool(results, "robust_pass"),
        "portfolio_eligible": _sum_bool(results, "portfolio_eligible"),
        "statistical_pass": _sum_bool(results, "statistical_pass"),
        "data_quality_pass": _sum_bool(results, "data_quality_pass"),
        "multiple_testing_pass": _sum_bool(results, "multiple_testing_pass"),
        "cost_pass": _sum_bool(results, "cost_pass"),
        "bootstrap_samples": int(args.bootstrap_samples),
        "locked_opened": False,
        "methods": _json_safe(methods.to_dict(orient="records")),
    }
    (output / "universal_robustness_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


def _sum_bool(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def _method_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "source_method" not in results.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for method, group in results.groupby("source_method", dropna=False):
        rows.append(
            {
                "method": str(method),
                "candidates": int(len(group)),
                "robust_pass": _sum_bool(group, "robust_pass"),
                "portfolio_eligible": _sum_bool(group, "portfolio_eligible"),
                "statistical_pass": _sum_bool(group, "statistical_pass"),
                "data_quality_pass": _sum_bool(group, "data_quality_pass"),
                "multiple_testing_pass": _sum_bool(group, "multiple_testing_pass"),
                "cost_pass": _sum_bool(group, "cost_pass"),
                "best_cagr": _max_numeric(group, "cagr"),
                "best_sharpe": _max_numeric(group, "sharpe"),
                "worst_mdd": _min_numeric(group, "mdd"),
            }
        )
    return pd.DataFrame(rows).sort_values(["robust_pass", "portfolio_eligible", "candidates"], ascending=False)


def _max_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").max())


def _min_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").min())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
