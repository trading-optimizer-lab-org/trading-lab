from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.config import load_yaml  # noqa: E402
from trading_lab.data_loader import load_market_data  # noqa: E402
from trading_lab.monthly_multi_asset import (  # noqa: E402
    MonthlyRiskSearchConfig,
    build_monthly_multi_asset_examples,
    run_monthly_multi_asset_search,
    write_monthly_multi_asset_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one monthly multi-asset risk search stage.")
    parser.add_argument("--config", default="configs/monthly_multi_asset.yaml")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--total-stages", type=int, default=64)
    parser.add_argument("--seed-pool", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--mutations-per-parent", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--top-rows-per-stage", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path("outputs/monthly_multi_asset")
    rows = []
    examples_count = 0
    try:
        raw_config = load_yaml(args.config)
        data_path = raw_config.get("data_path", "data/public/spy_daily.csv")
        output_dir = Path(args.output_dir or raw_config.get("output_dir", "outputs/monthly_multi_asset"))
        daily = load_market_data(data_path)
        examples = build_monthly_multi_asset_examples(
            daily,
            start_year=int(raw_config.get("start_year", 1994)),
            end_year=int(raw_config.get("end_year", 2026)),
        )
        examples_count = len(examples)
        config = MonthlyRiskSearchConfig(
            stage=args.stage,
            total_stages=args.total_stages,
            seed_pool=int(args.seed_pool or raw_config.get("seed_pool", 600)),
            beam_width=int(args.beam_width or raw_config.get("beam_width", 40)),
            generations=int(args.generations or raw_config.get("generations", 6)),
            mutations_per_parent=int(args.mutations_per_parent or raw_config.get("mutations_per_parent", 12)),
            max_features=int(args.max_features or raw_config.get("max_features", 6)),
            random_seed=int(args.random_seed or raw_config.get("random_seed", 612_000)),
            top_rows_per_stage=int(args.top_rows_per_stage or raw_config.get("top_rows_per_stage", 250)),
        )
        rows = run_monthly_multi_asset_search(examples, config)
        write_monthly_multi_asset_outputs(rows, examples, output_dir, stage=args.stage)
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "candidate_id": f"monthly_multi_asset_stage_error_{args.stage}",
                "stage": args.stage,
                "stage_failed": True,
                "stage_error": str(exc),
                "monthly_multi_asset_score": -1_000_000.0,
                "accepted": False,
                "locked_opened": False,
            }
        ]
        (output_dir / "monthly_multi_asset_summary.json").write_text(
            json.dumps(
                {
                    "stage": args.stage,
                    "stage_failed": True,
                    "stage_error": str(exc),
                    "candidates_evaluated": 0,
                    "accepted": 0,
                    "locked_opened": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "rows": len(rows),
                "examples": examples_count,
                "accepted": int(sum(bool(row.get("accepted")) for row in rows)),
                "stage_failed": any(bool(row.get("stage_failed")) for row in rows),
                "best_candidate": rows[0].get("candidate_id") if rows else None,
                "best_train_min_year_return": rows[0].get("train_min_year_return") if rows else None,
                "best_validation_min_year_return": rows[0].get("validation_min_year_return") if rows else None,
                "best_assets": rows[0].get("assets") if rows else None,
                "best_asset_selector": rows[0].get("asset_selector") if rows else None,
                "output_dir": str(output_dir),
                "locked_opened": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
