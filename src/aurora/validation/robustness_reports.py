from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurora.validation.robustness_config import UniversalRobustnessConfig


OUTPUT_FILES = (
    "universal_robustness_results.csv",
    "universal_robustness_pass.csv",
    "universal_robustness_methods.csv",
    "universal_robustness_summary.json",
    "universal_robustness_year_by_year.csv",
    "universal_robustness_fail_reasons.csv",
    "universal_robustness_config.json",
    "universal_robustness_duplicates.csv",
    "universal_robustness_bootstrap.csv",
    "universal_robustness_data_quality.csv",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_universal_robustness_outputs(
    output_dir: str | Path,
    *,
    results: pd.DataFrame,
    pass_results: pd.DataFrame | None = None,
    methods: pd.DataFrame | None = None,
    year_by_year: pd.DataFrame | None = None,
    fail_reasons: pd.DataFrame | None = None,
    duplicates: pd.DataFrame | None = None,
    bootstrap: pd.DataFrame | None = None,
    data_quality: pd.DataFrame | None = None,
    config: UniversalRobustnessConfig | None = None,
) -> dict[str, str]:
    """Write the standard universal robustness artifact bundle."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    if pass_results is None:
        pass_results = (
            results[results["robust_pass"].fillna(False).astype(bool)].copy()
            if "robust_pass" in results.columns
            else pd.DataFrame()
        )
    methods = methods if methods is not None else _method_summary(results)
    year_by_year = year_by_year if year_by_year is not None else pd.DataFrame()
    fail_reasons = fail_reasons if fail_reasons is not None else _fail_reason_table(results)
    duplicates = duplicates if duplicates is not None else pd.DataFrame()
    bootstrap = bootstrap if bootstrap is not None else pd.DataFrame()
    data_quality = data_quality if data_quality is not None else pd.DataFrame()
    config = config or UniversalRobustnessConfig()

    frames = {
        "universal_robustness_results.csv": results,
        "universal_robustness_pass.csv": pass_results,
        "universal_robustness_methods.csv": methods,
        "universal_robustness_year_by_year.csv": year_by_year,
        "universal_robustness_fail_reasons.csv": fail_reasons,
        "universal_robustness_duplicates.csv": duplicates,
        "universal_robustness_bootstrap.csv": bootstrap,
        "universal_robustness_data_quality.csv": data_quality,
    }
    written: dict[str, str] = {}
    for name, frame in frames.items():
        destination = path / name
        frame.to_csv(destination, index=False)
        written[name] = str(destination)

    summary = {
        "rows": int(len(results)),
        "robust_pass": int(results.get("robust_pass", pd.Series(dtype=bool)).fillna(False).sum()),
        "portfolio_eligible": int(
            results.get("portfolio_eligible", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "methods": json_safe(methods.to_dict(orient="records")),
    }
    summary_path = path / "universal_robustness_summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    written["universal_robustness_summary.json"] = str(summary_path)

    config_path = path / "universal_robustness_config.json"
    config_path.write_text(json.dumps(json_safe(config.to_dict()), indent=2, sort_keys=True), encoding="utf-8")
    written["universal_robustness_config.json"] = str(config_path)
    return written


def _method_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "source_method" not in results.columns:
        return pd.DataFrame(columns=["source_method", "candidates", "robust_pass", "portfolio_eligible"])
    rows: list[dict[str, object]] = []
    for method, group in results.groupby("source_method", dropna=False):
        rows.append(
            {
                "source_method": str(method),
                "candidates": int(len(group)),
                "robust_pass": int(group.get("robust_pass", pd.Series(dtype=bool)).fillna(False).sum()),
                "portfolio_eligible": int(
                    group.get("portfolio_eligible", pd.Series(dtype=bool)).fillna(False).sum()
                ),
                "best_cagr": float(group["cagr"].max()) if "cagr" in group.columns and len(group) else np.nan,
                "best_sharpe": float(group["sharpe"].max()) if "sharpe" in group.columns and len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _fail_reason_table(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if results.empty:
        return pd.DataFrame(columns=["candidate_id", "fail_reason"])
    for _, row in results.iterrows():
        reasons = str(row.get("fail_reasons", "") or "")
        for reason in [part for part in reasons.split(";") if part]:
            rows.append({"candidate_id": row.get("candidate_id", ""), "fail_reason": reason})
    return pd.DataFrame(rows, columns=["candidate_id", "fail_reason"])
