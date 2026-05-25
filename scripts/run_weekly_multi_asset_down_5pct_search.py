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
from trading_lab.weekly_multi_asset import (  # noqa: E402
    WEEKLY_METHODS,
    build_weekly_multi_asset_examples,
    run_weekly_multi_asset_search,
    write_weekly_multi_asset_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one weekly multi-asset SP500 down-year search stage.")
    parser.add_argument("--config", default="configs/weekly_multi_asset_sp500_down_5pct.yaml")
    parser.add_argument("--method", required=True, choices=WEEKLY_METHODS)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--total-stages", type=int, default=100)
    parser.add_argument("--seed-pool", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--mutations-per-parent", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--top-rows-per-stage", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path("outputs/weekly_multi_asset_sp500_down_5pct")
    rows = []
    examples_count = 0
    try:
        raw_config = load_yaml(args.config)
        data_path = raw_config.get("data_path", "data/public/spy_daily.csv")
        output_dir = Path(args.output_dir or raw_config.get("output_dir", "outputs/weekly_multi_asset_sp500_down_5pct"))
        daily = load_market_data(data_path)
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
            top_rows_per_stage=int(args.top_rows_per_stage or raw_config.get("top_rows_per_stage", 1200)),
        )
        rows = run_weekly_multi_asset_search(examples, config, method=args.method)
        write_weekly_multi_asset_outputs(rows, examples, output_dir, method=args.method, stage=args.stage)
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "candidate_id": f"weekly_multi_asset_stage_error_{args.method}_{args.stage}",
                "method": args.method,
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
        (output_dir / "weekly_multi_asset_sp500_down_5pct_leaderboard.csv").write_text(
            "candidate_id,method,stage,stage_failed,stage_error,weekly_multi_asset_score,accepted,locked_opened,validation_role\n"
            + f"weekly_multi_asset_stage_error_{args.method}_{args.stage},{args.method},{args.stage},true,{json.dumps(str(exc))},-1000000,false,false,report_only\n",
            encoding="utf-8",
        )
        (output_dir / f"weekly_multi_asset_sp500_down_5pct_leaderboard_stage_{args.method}_{args.stage}.csv").write_text(
            (output_dir / "weekly_multi_asset_sp500_down_5pct_leaderboard.csv").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (output_dir / "weekly_multi_asset_sp500_down_5pct_summary.json").write_text(
            json.dumps(
                {
                    "method": args.method,
                    "stage": args.stage,
                    "stage_failed": True,
                    "stage_error": str(exc),
                    "candidates_evaluated": 0,
                    "accepted": 0,
                    "verified_train_validation_5pct": 0,
                    "locked_opened": False,
                    "validation_role": "report_only",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "method": args.method,
                "stage": args.stage,
                "rows": len(rows),
                "examples": examples_count,
                "accepted": int(sum(bool(row.get("accepted")) for row in rows)),
                "verified_train_validation_5pct": int(sum(bool(row.get("verified_train_validation_5pct")) for row in rows)),
                "stage_failed": any(bool(row.get("stage_failed")) for row in rows),
                "best_candidate": rows[0].get("candidate_id") if rows else None,
                "best_train_min_year_return": rows[0].get("train_min_year_return") if rows else None,
                "best_validation_min_year_return": rows[0].get("validation_min_year_return") if rows else None,
                "best_train_down_min_return": rows[0].get("train_down_min_return") if rows else None,
                "best_validation_down_min_return": rows[0].get("validation_down_min_return") if rows else None,
                "best_assets": rows[0].get("assets") if rows else None,
                "best_asset_selector": rows[0].get("asset_selector") if rows else None,
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
