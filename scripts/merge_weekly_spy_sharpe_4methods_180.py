from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Iterable, Any

import pandas as pd


DEFAULT_METHODS = ("beam", "genetic", "aurora_ml", "github_ml")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge weekly SPY Sharpe 4-method 180-parallel stage outputs.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", default="outputs/weekly_spy_sharpe_4methods_180")
    parser.add_argument("--file-prefix", default="weekly_spy_sharpe_4methods_180")
    parser.add_argument("--expected-jobs", type=int, default=180)
    args = parser.parse_args()
    summary = merge_outputs(
        input_glob=args.input_glob,
        output_dir=args.output_dir,
        file_prefix=args.file_prefix,
        expected_jobs=args.expected_jobs,
        expected_methods=DEFAULT_METHODS,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def merge_outputs(
    *,
    input_glob: str,
    output_dir: str | Path,
    file_prefix: str,
    expected_jobs: int,
    expected_methods: Iterable[str] = DEFAULT_METHODS,
    score_mode: str = "train_sharpe_max_validation_80pct_report",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(input_glob, recursive=True))
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    leaderboard = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not leaderboard.empty:
        leaderboard = _stamp_validity(leaderboard, score_mode=score_mode)
        leaderboard = leaderboard.sort_values("weekly_multi_asset_score", ascending=False, na_position="last").reset_index(drop=True)
    else:
        leaderboard = _empty_leaderboard()
    verified = leaderboard.loc[leaderboard["verified_sharpe_robust"].astype(bool)].copy()
    relaxed = leaderboard.loc[leaderboard["relaxed_valid"].astype(bool)].copy()
    method_summary = _method_summary(leaderboard, verified, relaxed, expected_methods)
    efficiency = _efficiency(leaderboard, verified, relaxed, expected_methods)
    parallelism = _parallelism_summary(_meta_root_from_glob(input_glob), expected_jobs=expected_jobs, expected_methods=expected_methods)

    leaderboard.to_csv(output / f"{file_prefix}_leaderboard.csv", index=False)
    verified.to_csv(output / f"{file_prefix}_verified.csv", index=False)
    relaxed.to_csv(output / f"{file_prefix}_relaxed_valid.csv", index=False)
    method_summary.to_csv(output / f"{file_prefix}_methods.csv", index=False)
    efficiency.to_csv(output / f"{file_prefix}_efficiency.csv", index=False)
    (output / f"{file_prefix}_parallelism.json").write_text(json.dumps(parallelism, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "artifact": "weekly-spy-sharpe-4methods-2h-fair-180-parallel-leaderboard",
        "rows": int(len(leaderboard)),
        "input_files": int(len(paths)),
        "verified_sharpe_robust": int(len(verified)),
        "unique_verified_sharpe_robust": int(verified["candidate_id"].nunique()) if "candidate_id" in verified else 0,
        "relaxed_valid": int(len(relaxed)),
        "unique_relaxed_valid": int(relaxed["candidate_id"].nunique()) if "candidate_id" in relaxed else 0,
        "best_candidate": str(leaderboard.iloc[0].get("candidate_id", "")) if not leaderboard.empty else None,
        "best_method": str(leaderboard.iloc[0].get("method", "")) if not leaderboard.empty else None,
        "best_train_sharpe": _first_float(leaderboard, "train_sharpe"),
        "best_validation_sharpe": _first_float(leaderboard, "validation_sharpe"),
        "jobs_started": int(parallelism["jobs_started"]),
        "jobs_completed": int(parallelism["jobs_completed"]),
        "expected_jobs": int(expected_jobs),
        "max_parallel_observed": int(parallelism["max_parallel_observed"]),
        "parallelism_valid": bool(parallelism["parallelism_valid"]),
        "partial": int(parallelism["jobs_completed"]) < int(expected_jobs),
        "locked_opened": bool(_locked_opened(leaderboard)),
        "validation_role": "report_only",
        "score_mode": score_mode,
        "methods": list(expected_methods),
    }
    (output / f"{file_prefix}_summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _stamp_validity(frame: pd.DataFrame, *, score_mode: str = "train_sharpe_max_validation_80pct_report") -> pd.DataFrame:
    out = frame.copy()
    for column in (
        "train_sharpe",
        "validation_sharpe",
        "weekly_multi_asset_score",
        "train_cagr",
        "average_abs_exposure",
        "train_years_positive",
    ):
        out[column] = pd.to_numeric(out.get(column, pd.Series(dtype=float)), errors="coerce")
    locked = out.get("locked_opened", pd.Series(False, index=out.index)).astype(str).str.lower().isin(("true", "1", "yes"))
    ratio = out["validation_sharpe"] / out["train_sharpe"].where(out["train_sharpe"] > 0)
    out["validation_sharpe_ratio_to_train"] = ratio
    out["train_sharpe_gt_1"] = out["train_sharpe"] > 1.0
    out["validation_sharpe_gt_1"] = out["validation_sharpe"] > 1.0
    out["validation_sharpe_ge_80pct_train"] = ratio >= 0.80
    if score_mode == "train_sharpe_positive_years_report_validation":
        out["verified_sharpe_robust"] = (
            out["train_sharpe_gt_1"]
            & (out["train_cagr"] >= 0.04)
            & (out["train_years_positive"] >= 10)
            & (out["average_abs_exposure"] >= 0.15)
            & ~locked
        )
        out["relaxed_valid"] = (out["train_sharpe"] > 0.0) & ~locked
    else:
        out["verified_sharpe_robust"] = (
            out["train_sharpe_gt_1"]
            & out["validation_sharpe_gt_1"]
            & out["validation_sharpe_ge_80pct_train"]
            & ~locked
        )
        out["relaxed_valid"] = (out["train_sharpe"] > 0.0) & (out["validation_sharpe"] > 0.0) & ~locked
    out["locked_opened"] = locked
    return out


def _method_summary(rows: pd.DataFrame, verified: pd.DataFrame, relaxed: pd.DataFrame, expected_methods: Iterable[str]) -> pd.DataFrame:
    records = []
    for method in expected_methods:
        group = rows.loc[rows.get("method", pd.Series(dtype=str)).astype(str) == method] if not rows.empty else pd.DataFrame()
        method_verified = verified.loc[verified.get("method", pd.Series(dtype=str)).astype(str) == method] if not verified.empty else pd.DataFrame()
        method_relaxed = relaxed.loc[relaxed.get("method", pd.Series(dtype=str)).astype(str) == method] if not relaxed.empty else pd.DataFrame()
        records.append(
            {
                "method": method,
                "rows": int(len(group)),
                "jobs_completed": _unique_stage_count(group),
                "verified_count": int(len(method_verified)),
                "relaxed_valid_count": int(len(method_relaxed)),
                "best_candidate": str(group.iloc[0].get("candidate_id", "")) if not group.empty else "",
                "best_train_sharpe": _best_float(group, "train_sharpe"),
                "best_validation_sharpe": _best_float(group, "validation_sharpe"),
                "best_train_years_positive": _best_float(group, "train_years_positive"),
                "best_validation_years_positive": _best_float(group, "validation_years_positive"),
                "best_score": _best_float(group, "weekly_multi_asset_score"),
            }
        )
    return pd.DataFrame(records)


def _efficiency(rows: pd.DataFrame, verified: pd.DataFrame, relaxed: pd.DataFrame, expected_methods: Iterable[str]) -> pd.DataFrame:
    records = []
    for method in expected_methods:
        group = rows.loc[rows.get("method", pd.Series(dtype=str)).astype(str) == method] if not rows.empty else pd.DataFrame()
        method_verified = verified.loc[verified.get("method", pd.Series(dtype=str)).astype(str) == method] if not verified.empty else pd.DataFrame()
        method_relaxed = relaxed.loc[relaxed.get("method", pd.Series(dtype=str)).astype(str) == method] if not relaxed.empty else pd.DataFrame()
        elapsed = _best_float(group, "elapsed_seconds") or 0.0
        hours = max(float(elapsed) / 3600.0, 1e-9)
        first_verified = None
        if not method_verified.empty and "first_seen_minute" in method_verified:
            first_verified = float(pd.to_numeric(method_verified["first_seen_minute"], errors="coerce").min())
        records.append(
            {
                "method": method,
                "rows": int(len(group)),
                "verified_count": int(len(method_verified)),
                "relaxed_valid_count": int(len(method_relaxed)),
                "verified_per_hour": float(len(method_verified) / hours),
                "relaxed_valid_per_hour": float(len(method_relaxed) / hours),
                "time_to_first_verified": first_verified,
            }
        )
    return pd.DataFrame(records)


def _parallelism_summary(root: Path, *, expected_jobs: int, expected_methods: Iterable[str]) -> dict[str, Any]:
    metas = []
    for path in sorted(root.rglob("job_meta.json")) if root.exists() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("started_epoch") is None:
            continue
        metas.append(payload)
    events = []
    for meta in metas:
        start = float(meta.get("started_epoch") or 0.0)
        end = float(meta.get("ended_epoch") or start)
        events.append((start, 1))
        events.append((end, -1))
    active = 0
    max_active = 0
    for _, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        active += delta
        max_active = max(max_active, active)
    started_by_method = {method: 0 for method in expected_methods}
    completed_by_method = {method: 0 for method in expected_methods}
    for meta in metas:
        method = str(meta.get("method", ""))
        if method in started_by_method:
            started_by_method[method] += 1
            if meta.get("ended_epoch") is not None:
                completed_by_method[method] += 1
    starts = [float(meta.get("started_epoch")) for meta in metas if meta.get("started_epoch") is not None]
    return {
        "jobs_started": int(len(metas)),
        "jobs_completed": int(sum(1 for meta in metas if meta.get("ended_epoch") is not None)),
        "expected_jobs": int(expected_jobs),
        "max_parallel_observed": int(max_active),
        "parallelism_valid": bool(max_active >= int(expected_jobs)),
        "first_start_epoch": min(starts) if starts else None,
        "last_start_epoch": max(starts) if starts else None,
        "first_last_start_spread_seconds": (max(starts) - min(starts)) if starts else None,
        "started_by_method": started_by_method,
        "completed_by_method": completed_by_method,
    }


def _meta_root_from_glob(pattern: str) -> Path:
    marker = "**"
    if marker in pattern:
        return Path(pattern.split(marker, 1)[0])
    return Path(pattern).parent


def _empty_leaderboard() -> pd.DataFrame:
    return pd.DataFrame(columns=["candidate_id", "method", "stage", "weekly_multi_asset_score", "train_sharpe", "validation_sharpe"])


def _locked_opened(frame: pd.DataFrame) -> bool:
    if frame.empty or "locked_opened" not in frame:
        return False
    return bool(frame["locked_opened"].astype(bool).any())


def _unique_stage_count(frame: pd.DataFrame) -> int:
    if frame.empty or "method" not in frame or "stage" not in frame:
        return 0
    if "wave" in frame:
        return int(frame[["method", "wave", "stage"]].drop_duplicates().shape[0])
    return int(frame[["method", "stage"]].drop_duplicates().shape[0])


def _best_float(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else None


def _first_float(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    value = pd.to_numeric(pd.Series([frame.iloc[0].get(column)]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
