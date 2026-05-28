from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_weekly_spy_sharpe_4methods_180_stage import (  # noqa: E402
    _run_aurora_stage,
    _spy_common_daily,
    _write_job_meta,
)
from trading_lab.config import load_yaml  # noqa: E402
from trading_lab.data_loader import load_market_data  # noqa: E402
from trading_lab.monthly_risk import MonthlyRiskSearchConfig  # noqa: E402
from trading_lab.weekly_7methods_stateful import (  # noqa: E402
    ALL_STATEFUL_WEEKLY_METHODS,
    build_method_state,
    run_stateful_weekly_search,
    write_stateful_weekly_outputs,
)
from trading_lab.weekly_multi_asset import WEEKLY_MAX_SHARPE_SCORE_MODE, build_weekly_multi_asset_examples  # noqa: E402


METHODS = (
    "beam",
    "genetic",
    "sobol_random_asha_real",
    "optuna_tpe_hyperband",
    "dehb_real",
    "bohb_real",
    "smac_mf_real",
    "bandit",
    "aurora_ml",
    "github_ml",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one SPY Sharpe 10-method overnight stage.")
    parser.add_argument("--config", default="configs/weekly_spy_sharpe_10methods_9h_waves.yaml")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--total-stages", type=int, default=18)
    parser.add_argument("--time-budget-minutes", type=float, default=85.0)
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--file-prefix", default="weekly_spy_sharpe_10methods_9h")
    parser.add_argument("--top-rows-per-stage", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    started_epoch = time.time()
    rows: list[dict[str, Any]] = []
    state: dict[str, object] = {}
    try:
        raw_config = load_yaml(args.config)
        daily = _spy_common_daily(load_market_data(raw_config.get("data_path", "data/public/spy_daily.csv")))
        if args.method == "aurora_ml":
            rows = _run_aurora_stage(daily, raw_config, args, output_dir)
            state = _simple_state(rows, method=args.method, wave=args.wave, stage=args.stage)
        else:
            rows, state = _run_stateful_stage(daily, raw_config, args)
        write_stateful_weekly_outputs(
            rows,
            state,
            output_dir,
            method=args.method,
            wave=args.wave,
            stage=args.stage,
            file_prefix=args.file_prefix,
        )
    except Exception as exc:
        rows = [_error_row(args, exc, score_mode=_score_mode_from_config(args.config))]
        state = _simple_state(rows, method=args.method, wave=args.wave, stage=args.stage)
        write_stateful_weekly_outputs(
            rows,
            state,
            output_dir,
            method=args.method,
            wave=args.wave,
            stage=args.stage,
            file_prefix=args.file_prefix,
        )
    finally:
        _write_job_meta(output_dir, args, started_epoch=started_epoch, started_iso=_utc_now(), rows=len(rows))
    print(json.dumps({"method": args.method, "wave": args.wave, "stage": args.stage, "rows": len(rows), "output_dir": str(output_dir)}, indent=2))
    return 0


def _run_stateful_stage(daily, raw_config: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    examples = build_weekly_multi_asset_examples(
        daily,
        benchmark_daily=None,
        start_year=int(raw_config.get("start_year", 1994)),
        end_year=int(raw_config.get("end_year", 2026)),
    )
    engine_method = "machine_learning" if args.method == "github_ml" else args.method
    if engine_method not in ALL_STATEFUL_WEEKLY_METHODS:
        raise ValueError(f"unknown 10-method engine: {engine_method}")
    search_config = MonthlyRiskSearchConfig(
        stage=int(args.stage),
        total_stages=int(args.total_stages),
        seed_pool=int(raw_config.get("seed_pool", 5000)),
        beam_width=int(raw_config.get("beam_width", 160)),
        generations=int(raw_config.get("generations", 24)),
        mutations_per_parent=int(raw_config.get("mutations_per_parent", 36)),
        max_features=int(raw_config.get("max_features", 8)),
        random_seed=int(args.random_seed or raw_config.get("random_seed", 1210900)),
        top_rows_per_stage=int(args.top_rows_per_stage),
        score_mode=str(raw_config.get("score_mode", WEEKLY_MAX_SHARPE_SCORE_MODE)),
    )
    rows, state = run_stateful_weekly_search(
        examples,
        search_config,
        method=engine_method,
        wave=int(args.wave),
        stage=int(args.stage),
        time_budget_minutes=float(args.time_budget_minutes),
        state_dir=args.state_dir,
    )
    if args.method == "github_ml":
        for row in rows:
            row["method"] = "github_ml"
            row["candidate_id"] = str(row.get("candidate_id", "")).replace("machine_learning", "github_ml")
            row["locked_opened"] = False
            row["validation_role"] = "report_only"
        state = build_method_state(rows, method="github_ml", wave=int(args.wave), stage=int(args.stage))
    return rows, state


def _simple_state(rows: list[dict[str, Any]], *, method: str, wave: int, stage: int) -> dict[str, object]:
    candidates = []
    for row in sorted(rows, key=lambda item: float(item.get("weekly_multi_asset_score", 0.0) or 0.0), reverse=True)[:500]:
        candidates.append(
            {
                "candidate_id": str(row.get("candidate_id", "")),
                "features": str(row.get("features", "")),
                "model": str(row.get("ml_model", "")),
                "train_score": float(row.get("weekly_multi_asset_score", 0.0) or 0.0),
                "train_sharpe": float(row.get("train_sharpe", 0.0) or 0.0),
                "validation_role": "report_only",
                "locked_opened": False,
            }
        )
    return {
        "method": method,
        "wave": int(wave),
        "stage": int(stage),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "validation_role": "report_only",
        "locked_opened": False,
    }


def _error_row(args: argparse.Namespace, exc: Exception, *, score_mode: str = WEEKLY_MAX_SHARPE_SCORE_MODE) -> dict[str, Any]:
    return {
        "candidate_id": f"weekly_spy_sharpe_10methods_stage_error_{args.method}_{args.wave}_{args.stage}",
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
        "score_mode": score_mode,
    }


def _score_mode_from_config(config_path: str) -> str:
    try:
        raw_config = load_yaml(config_path)
    except Exception:
        return WEEKLY_MAX_SHARPE_SCORE_MODE
    return str(raw_config.get("score_mode", WEEKLY_MAX_SHARPE_SCORE_MODE))


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
