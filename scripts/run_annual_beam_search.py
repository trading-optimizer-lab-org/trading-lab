from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.annual_prediction import AnnualBeamConfig, build_annual_examples, run_annual_beam_search  # noqa: E402
from trading_lab.config import load_yaml  # noqa: E402
from trading_lab.data_loader import load_market_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one annual SP500 Beam prediction stage.")
    parser.add_argument("--config", default="configs/annual_sp500_beam.yaml")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--total-stages", type=int, default=64)
    parser.add_argument("--seed-pool", type=int, default=500)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--mutations-per-parent", type=int, default=12)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--score-mode", default=None)
    parser.add_argument("--top-rows-per-stage", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path("outputs/annual_sp500_beam")
    examples_count = 0
    try:
        raw_config = load_yaml(args.config)
        data_path = raw_config.get("data_path", "data/public/sp500_annual_daily.csv")
        output_dir = Path(raw_config.get("output_dir", "outputs/annual_sp500_beam"))
        daily = load_market_data(data_path)
        examples = build_annual_examples(
            daily,
            start_year=int(raw_config.get("start_year", 1980)),
            end_year=int(raw_config.get("end_year", 2025)),
        )
        examples_count = len(examples)
        config = AnnualBeamConfig(
            stage=args.stage,
            total_stages=args.total_stages,
            seed_pool=args.seed_pool,
            beam_width=args.beam_width,
            generations=args.generations,
            mutations_per_parent=args.mutations_per_parent,
            max_features=int(args.max_features or raw_config.get("max_features", 4)),
            score_mode=str(args.score_mode or raw_config.get("score_mode", "validation")),
        )
        rows = run_annual_beam_search(examples, config)
    except Exception as exc:
        rows = [
            {
                "candidate_id": f"annual_beam_stage_error_{args.stage}",
                "stage": args.stage,
                "stage_failed": True,
                "stage_error": str(exc),
                "annual_score": -1_000_000.0,
                "accepted": False,
                "locked_opened": False,
            }
        ]
    total_rows = len(rows)
    for row in rows:
        row["stage"] = args.stage
        row["stage_candidates_evaluated"] = total_rows
    top_rows = int(args.top_rows_per_stage or 0)
    if top_rows > 0 and len(rows) > top_rows:
        accepted_rows = [row for row in rows if bool(row.get("accepted"))]
        top = sorted(rows, key=lambda row: float(row.get("annual_score", -1_000_000.0) or -1_000_000.0), reverse=True)[
            :top_rows
        ]
        by_id = {str(row.get("candidate_id")): row for row in [*top, *accepted_rows]}
        rows = sorted(by_id.values(), key=lambda row: float(row.get("annual_score", -1_000_000.0) or -1_000_000.0), reverse=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"annual_sp500_beam_stage_{args.stage}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "rows": len(rows),
                "candidates_evaluated": total_rows,
                "examples": examples_count,
                "accepted": int(sum(bool(row.get("accepted")) for row in rows)),
                "stage_failed": any(bool(row.get("stage_failed")) for row in rows),
                "best_accuracy_validation": max((float(row.get("validation_accuracy", 0.0) or 0.0) for row in rows), default=0.0),
                "best_accuracy_train": max((float(row.get("train_accuracy", 0.0) or 0.0) for row in rows), default=0.0),
                "output_path": str(output_path),
                "locked_opened": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
