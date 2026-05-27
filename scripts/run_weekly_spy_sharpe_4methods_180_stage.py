from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aurora.data_contracts.timeseries_store import TimeSeriesStore  # noqa: E402
from aurora.research.ml_search import MLSearchConfig, run_ml_search  # noqa: E402
from trading_lab.config import load_yaml  # noqa: E402
from trading_lab.data_loader import load_market_data  # noqa: E402
from trading_lab.monthly_risk import MonthlyRiskSearchConfig  # noqa: E402
from trading_lab.weekly_7methods_stateful import run_stateful_weekly_search, write_stateful_weekly_outputs  # noqa: E402
from trading_lab.weekly_multi_asset import WEEKLY_MAX_SHARPE_SCORE_MODE, build_weekly_multi_asset_examples  # noqa: E402


METHODS = ("beam", "genetic", "aurora_ml", "github_ml")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one SPY Sharpe 4-method 180-parallel stage.")
    parser.add_argument("--config", default="configs/weekly_spy_sharpe_4methods_180.yaml")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--wave", type=int, default=1)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--total-stages", type=int, default=45)
    parser.add_argument("--time-budget-minutes", type=float, default=100.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--file-prefix", default="weekly_spy_sharpe_4methods_180")
    parser.add_argument("--top-rows-per-stage", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    started_epoch = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    try:
        raw_config = load_yaml(args.config)
        daily = _spy_common_daily(load_market_data(raw_config.get("data_path", "data/public/spy_daily.csv")))
        if args.method == "aurora_ml":
            rows = _run_aurora_stage(daily, raw_config, args, output_dir)
            _write_stage_outputs(rows, output_dir, file_prefix=args.file_prefix, method=args.method, wave=args.wave, stage=args.stage)
        else:
            rows = _run_weekly_stage(daily, raw_config, args)
            write_stateful_weekly_outputs(rows, {}, output_dir, method=args.method, wave=args.wave, stage=args.stage, file_prefix=args.file_prefix)
    except Exception as exc:
        rows = [_error_row(args, exc)]
        _write_stage_outputs(rows, output_dir, file_prefix=args.file_prefix, method=args.method, wave=args.wave, stage=args.stage)
    finally:
        _write_job_meta(output_dir, args, started_epoch=started_epoch, started_iso=started_iso, rows=len(rows))
    print(json.dumps({"method": args.method, "stage": args.stage, "rows": len(rows), "output_dir": str(output_dir)}, indent=2))
    return 0


def _run_weekly_stage(daily: pd.DataFrame, raw_config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    examples = build_weekly_multi_asset_examples(
        daily,
        benchmark_daily=None,
        start_year=int(raw_config.get("start_year", 1994)),
        end_year=int(raw_config.get("end_year", 2026)),
    )
    search_config = MonthlyRiskSearchConfig(
        stage=int(args.stage),
        total_stages=int(args.total_stages),
        seed_pool=int(raw_config.get("seed_pool", 5000)),
        beam_width=int(raw_config.get("beam_width", 160)),
        generations=int(raw_config.get("generations", 24)),
        mutations_per_parent=int(raw_config.get("mutations_per_parent", 36)),
        max_features=int(raw_config.get("max_features", 8)),
        random_seed=int(args.random_seed or raw_config.get("random_seed", 1201800)),
        top_rows_per_stage=int(args.top_rows_per_stage),
        score_mode=str(raw_config.get("score_mode", WEEKLY_MAX_SHARPE_SCORE_MODE)),
    )
    engine_method = "machine_learning" if args.method == "github_ml" else args.method
    rows, _ = run_stateful_weekly_search(
        examples,
        search_config,
        method=engine_method,
        wave=int(args.wave),
        stage=int(args.stage),
        time_budget_minutes=float(args.time_budget_minutes),
    )
    for row in rows:
        row["method"] = args.method
        row["candidate_id"] = str(row.get("candidate_id", "")).replace("machine_learning", "github_ml")
        row["locked_opened"] = False
        row["validation_role"] = "report_only"
    return rows


def _run_aurora_stage(daily: pd.DataFrame, raw_config: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    qf_data = output_dir / "qf_data"
    run_root = output_dir / "aurora_runs"
    os.environ["QF_DATA_DIR"] = str(qf_data)
    train_start = pd.Timestamp(str(raw_config.get("train_start", "1994-01-01")))
    daily = daily.loc[daily.index >= train_start].copy()
    TimeSeriesStore().put("prices_daily", "SPY", daily, version=f"spy_sharpe_stage_{args.stage}", replace=True)
    models = tuple(str(item) for item in raw_config.get("aurora_ml_models", ["logistic", "forest", "ridge", "corr"]))
    report = run_ml_search(
        MLSearchConfig(
            run_id=f"aurora_ml_stage_{args.stage}",
            symbol="SPY",
            library="prices_daily",
            train_end=str(raw_config.get("train_end", "2007-12-31")),
            validation_start=str(raw_config.get("validation_start", "2008-01-01")),
            validation_end=str(raw_config.get("validation_end", "2019-12-31")),
            locked_start=str(raw_config.get("locked_start", "2020-01-01")),
            target_calmar=-999.0,
            validation_target_calmar=None,
            workers=1,
            max_candidates=1_000_000,
            batch_size=24,
            seed=int(args.random_seed or raw_config.get("random_seed", 1201800)) + int(args.stage),
            time_limit_seconds=max(1.0, float(args.time_budget_minutes) * 60.0),
            models=models,
            run_root=str(run_root),
            top_n=max(int(args.top_rows_per_stage), 500),
            target_objective_count=999999,
            min_train_cagr=None,
            min_validation_cagr=None,
            max_train_validation_calmar_ratio=None,
            min_train_subperiod_calmar=-999.0,
            defer_robustness_until_basic_pass=True,
            no_costs=True,
            no_locked=True,
        )
    )
    candidates = _read_aurora_candidates(Path(report.output_dir) / "candidates.jsonl")
    if not candidates:
        candidates = [candidate.to_dict() for candidate in report.top]
    rows = [_aurora_candidate_to_row(candidate, args) for candidate in candidates]
    rows.sort(key=lambda row: float(row.get("weekly_multi_asset_score", -1e18)), reverse=True)
    return rows[: int(args.top_rows_per_stage)]


def _aurora_candidate_to_row(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    train = candidate.get("train_metrics") or {}
    validation = candidate.get("validation_metrics") or {}
    train_sharpe = _to_float(train.get("sharpe"))
    validation_sharpe = _to_float(validation.get("sharpe"))
    train_cagr = _to_float(train.get("cagr")) / 100.0
    validation_cagr = _to_float(validation.get("cagr")) / 100.0
    train_mdd = _to_float(train.get("mdd")) / 100.0
    validation_mdd = _to_float(validation.get("mdd")) / 100.0
    feature_set = candidate.get("feature_set") or []
    ratio = validation_sharpe / train_sharpe if train_sharpe > 0 else float("nan")
    score = train_sharpe * 1_000_000.0 + train_cagr * 100_000.0 - abs(train_mdd) * 1_000.0 - max(0, len(feature_set) - 5) * 10.0
    return {
        "candidate_id": f"aurora_{candidate.get('candidate_id', '')}",
        "method": "aurora_ml",
        "wave": int(args.wave),
        "stage": int(args.stage),
        "ml_model": candidate.get("model", ""),
        "features": "|".join(str(item) for item in feature_set),
        "feature_count": len(feature_set),
        "train_cagr": train_cagr,
        "validation_cagr": validation_cagr,
        "train_sharpe": train_sharpe,
        "validation_sharpe": validation_sharpe,
        "train_mdd": train_mdd,
        "validation_mdd": validation_mdd,
        "train_calmar": _to_float(train.get("calmar")),
        "validation_calmar": _to_float(validation.get("calmar")),
        "validation_sharpe_ratio_to_train": ratio,
        "train_sharpe_gt_1": bool(train_sharpe > 1.0),
        "validation_sharpe_gt_1": bool(validation_sharpe > 1.0),
        "validation_sharpe_ge_80pct_train": bool(ratio >= 0.80),
        "verified_sharpe_robust": bool(train_sharpe > 1.0 and validation_sharpe > 1.0 and ratio >= 0.80),
        "accepted": bool(train_sharpe > 1.0),
        "weekly_multi_asset_score": score,
        "elapsed_seconds": float(args.time_budget_minutes) * 60.0,
        "candidates_evaluated": 0,
        "first_seen_wave": int(args.wave),
        "first_seen_minute": float(args.time_budget_minutes),
        "locked_opened": False,
        "validation_role": "report_only",
        "score_mode": WEEKLY_MAX_SHARPE_SCORE_MODE,
    }


def _spy_common_daily(daily: pd.DataFrame) -> pd.DataFrame:
    cols = [column for column in ("open", "high", "low", "close", "volume") if column in daily]
    out = daily.loc[:, cols].copy()
    out["adj_close"] = out["close"]
    return out


def _write_stage_outputs(rows: list[dict[str, Any]], output_dir: Path, *, file_prefix: str, method: str, wave: int, stage: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"{file_prefix}_leaderboard_stage_{stage}.csv", index=False)
    frame.loc[frame.get("verified_sharpe_robust", pd.Series(dtype=bool)).astype(bool)].to_csv(
        output_dir / f"{file_prefix}_verified_stage_{stage}.csv",
        index=False,
    )
    summary = {
        "method": method,
        "wave": int(wave),
        "stage": int(stage),
        "rows": int(len(rows)),
        "verified_sharpe_robust": int(sum(bool(row.get("verified_sharpe_robust")) for row in rows)),
        "locked_opened": False,
        "validation_role": "report_only",
    }
    (output_dir / f"{file_prefix}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _write_job_meta(output_dir: Path, args: argparse.Namespace, *, started_epoch: float, started_iso: str, rows: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ended_epoch = time.time()
    payload = {
        "method": args.method,
        "stage": int(args.stage),
        "wave": int(args.wave),
        "started_epoch": float(started_epoch),
        "ended_epoch": float(ended_epoch),
        "started_at": started_iso,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(ended_epoch - started_epoch),
        "rows": int(rows),
    }
    (output_dir / "job_meta.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_aurora_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def _error_row(args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    return {
        "candidate_id": f"weekly_spy_sharpe_4methods_stage_error_{args.method}_{args.stage}",
        "method": args.method,
        "wave": int(args.wave),
        "stage": int(args.stage),
        "stage_failed": True,
        "stage_error": str(exc),
        "weekly_multi_asset_score": -1_000_000_000.0,
        "train_sharpe": float("nan"),
        "validation_sharpe": float("nan"),
        "accepted": False,
        "verified_sharpe_robust": False,
        "locked_opened": False,
        "validation_role": "report_only",
    }


def _to_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
