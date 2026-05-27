from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.config import load_yaml  # noqa: E402
from trading_lab.data_loader import load_market_data  # noqa: E402
from trading_lab.monthly_risk import MonthlyRiskSearchConfig  # noqa: E402
from trading_lab.public_data import download_yahoo_chart  # noqa: E402
from trading_lab.weekly_7methods_stateful import (  # noqa: E402
    ALL_STATEFUL_WEEKLY_METHODS,
    run_stateful_weekly_search,
    write_stateful_weekly_outputs,
)
from trading_lab.weekly_multi_asset import build_weekly_multi_asset_examples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one stateful weekly 7-method search shard.")
    parser.add_argument("--config", default="configs/weekly_multi_asset_sp500_down_5pct.yaml")
    parser.add_argument("--method", required=True, choices=ALL_STATEFUL_WEEKLY_METHODS)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--total-stages", type=int, default=70)
    parser.add_argument("--time-budget-minutes", type=float, default=240.0)
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--seed-pool", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--mutations-per-parent", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--top-rows-per-stage", type=int, default=500)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--file-prefix", default="weekly_7methods_12h")
    parser.add_argument("--random-seed", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"outputs/weekly_7methods_12h/{args.method}/wave_{args.wave}/stage_{args.stage}")
    rows = []
    state = {}
    examples_count = 0
    try:
        raw_config = load_yaml(args.config)
        daily = load_market_data(raw_config.get("data_path", "data/public/spy_daily.csv"))
        try:
            benchmark_daily = download_yahoo_chart("^GSPC")
        except Exception:
            benchmark_daily = None
        examples = build_weekly_multi_asset_examples(
            daily,
            benchmark_daily=benchmark_daily,
            start_year=int(raw_config.get("start_year", 1994)),
            end_year=int(raw_config.get("end_year", 2026)),
        )
        examples_count = len(examples)
        config = MonthlyRiskSearchConfig(
            stage=args.stage,
            total_stages=args.total_stages,
            seed_pool=int(args.seed_pool or raw_config.get("seed_pool", 5000)),
            beam_width=int(args.beam_width or raw_config.get("beam_width", 160)),
            generations=int(args.generations or raw_config.get("generations", 24)),
            mutations_per_parent=int(args.mutations_per_parent or raw_config.get("mutations_per_parent", 36)),
            max_features=int(args.max_features or raw_config.get("max_features", 8)),
            random_seed=int(args.random_seed or raw_config.get("random_seed", 817_000)),
            top_rows_per_stage=int(args.top_rows_per_stage),
            score_mode=str(raw_config.get("score_mode", "train_only_weekly_sp500_down_5pct")),
        )
        rows, state = run_stateful_weekly_search(
            examples,
            config,
            method=args.method,
            wave=args.wave,
            stage=args.stage,
            time_budget_minutes=args.time_budget_minutes,
            state_dir=args.state_dir,
        )
        write_stateful_weekly_outputs(rows, state, output_dir, method=args.method, wave=args.wave, stage=args.stage, file_prefix=args.file_prefix)
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "candidate_id": f"weekly_7methods_stage_error_{args.method}_{args.wave}_{args.stage}",
                "method": args.method,
                "wave": args.wave,
                "stage": args.stage,
                "stage_failed": True,
                "stage_error": str(exc),
                "weekly_multi_asset_score": -1_000_000.0,
                "accepted": False,
                "verified_train_validation_5pct": False,
                "locked_opened": False,
                "validation_role": "report_only",
            }
        ]
        write_stateful_weekly_outputs(
            rows,
            {"method": args.method, "candidates": []},
            output_dir,
            method=args.method,
            wave=args.wave,
            stage=args.stage,
            file_prefix=args.file_prefix,
        )
    print(
        json.dumps(
            {
                "method": args.method,
                "wave": args.wave,
                "stage": args.stage,
                "rows": len(rows),
                "examples": examples_count,
                "accepted": int(sum(bool(row.get("accepted")) for row in rows)),
                "verified_train_validation_5pct": int(sum(bool(row.get("verified_train_validation_5pct")) for row in rows)),
                "best_candidate": rows[0].get("candidate_id") if rows else None,
                "state_candidates": len(state.get("candidates", [])) if isinstance(state, dict) else 0,
                "output_dir": str(output_dir),
                "locked_opened": False,
                "validation_role": "report_only",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
