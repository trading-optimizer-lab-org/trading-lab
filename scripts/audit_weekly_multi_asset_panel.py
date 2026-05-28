from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.data_loader import load_market_data  # noqa: E402
from trading_lab.monthly_multi_asset import TRADABLE_ASSETS  # noqa: E402
from trading_lab.monthly_risk import _json_safe  # noqa: E402
from trading_lab.weekly_multi_asset import _available_assets_from_examples, build_weekly_multi_asset_examples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit weekly multi-asset public panel before search.")
    parser.add_argument("--data", default="data/public/spy_daily.csv")
    parser.add_argument("--output", default="outputs/weekly_multi_asset_panel_audit.json")
    parser.add_argument("--start-year", type=int, default=1994)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--min-assets", type=int, default=10)
    args = parser.parse_args()

    daily = load_market_data(args.data)
    examples = build_weekly_multi_asset_examples(
        daily,
        benchmark_daily=None,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
    )
    available_assets = _available_assets_from_examples(examples)
    expected_ratio_columns = [f"{asset.lower()}_close_ratio" for asset in TRADABLE_ASSETS if asset != "SPY"]
    present_ratio_columns = [column for column in expected_ratio_columns if column in daily.columns]
    audit = {
        "data": str(args.data),
        "rows_daily": int(len(daily)),
        "rows_weekly_examples": int(len(examples)),
        "tradable_assets_defined": list(TRADABLE_ASSETS),
        "available_assets": available_assets,
        "available_asset_count": int(len(available_assets)),
        "present_ratio_column_count": int(len(present_ratio_columns)),
        "present_ratio_columns": present_ratio_columns,
        "multi_asset_panel": bool(len(available_assets) >= int(args.min_assets)),
        "locked_opened": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(audit), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_json_safe(audit), indent=2, sort_keys=True))
    if not audit["multi_asset_panel"]:
        raise SystemExit(f"weekly panel has only {len(available_assets)} available assets; expected at least {args.min_assets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
