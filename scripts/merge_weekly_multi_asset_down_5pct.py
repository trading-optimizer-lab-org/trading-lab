from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.config import load_yaml  # noqa: E402
from trading_lab.data_loader import load_market_data  # noqa: E402
from trading_lab.public_data import download_yahoo_chart  # noqa: E402
from trading_lab.weekly_multi_asset import build_weekly_multi_asset_examples, merge_weekly_multi_asset_leaderboards  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge weekly multi-asset SP500 down-year leaderboards.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", default="outputs/weekly_multi_asset_sp500_down_5pct")
    parser.add_argument("--config", default="configs/weekly_multi_asset_sp500_down_5pct.yaml")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob, recursive=True))
    examples = None
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
    except Exception:
        examples = None
    summary = merge_weekly_multi_asset_leaderboards(paths, args.output_dir, examples=examples)
    print(json.dumps({"input_files": len(paths), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
