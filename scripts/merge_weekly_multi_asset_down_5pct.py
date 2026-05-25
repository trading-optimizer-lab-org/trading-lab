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
    parser.add_argument("--skip-examples", action="store_true", help="Skip rebuilding weekly examples and best-position tables.")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob, recursive=True))
    examples = None
    if not args.skip_examples:
        print("loading public panel for best-candidate position tables", flush=True)
        try:
            raw_config = load_yaml(args.config)
            daily = load_market_data(raw_config.get("data_path", "data/public/spy_daily.csv"))
            try:
                print("loading benchmark ^GSPC for down-year labels", flush=True)
                benchmark_daily = download_yahoo_chart("^GSPC")
            except Exception:
                benchmark_daily = None
            print("building weekly examples", flush=True)
            examples = build_weekly_multi_asset_examples(
                daily,
                benchmark_daily=benchmark_daily,
                start_year=int(raw_config.get("start_year", 1994)),
                end_year=int(raw_config.get("end_year", 2026)),
            )
            print(f"weekly examples: {len(examples)}", flush=True)
        except Exception as exc:
            print(f"warning: could not build examples: {exc}", flush=True)
            examples = None
    else:
        print("skipping weekly examples for partial merge", flush=True)
    print(f"input leaderboard files: {len(paths)}", flush=True)
    summary = merge_weekly_multi_asset_leaderboards(
        paths,
        args.output_dir,
        examples=examples,
        progress_every=args.progress_every,
    )
    print(json.dumps({"input_files": len(paths), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
