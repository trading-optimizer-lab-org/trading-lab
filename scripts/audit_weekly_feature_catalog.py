from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.config import load_yaml  # noqa: E402
from trading_lab.data_loader import load_market_data  # noqa: E402
from trading_lab.monthly_multi_asset import TRADABLE_ASSETS  # noqa: E402
from trading_lab.weekly_multi_asset import (  # noqa: E402
    _available_assets_from_examples,
    _build_weekly_spec_catalog,
    build_weekly_multi_asset_examples,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit weekly multi-asset feature catalog.")
    parser.add_argument("--config", default="configs/weekly_beam_genetic_2h_max_calmar.yaml")
    parser.add_argument("--output-dir", default="outputs/weekly_beam_genetic_2h_max_calmar")
    parser.add_argument("--file-prefix", default="weekly_beam_genetic_2h")
    args = parser.parse_args()

    raw_config = load_yaml(args.config)
    daily = load_market_data(raw_config.get("data_path", "data/public/spy_daily.csv"))
    examples = build_weekly_multi_asset_examples(
        daily,
        benchmark_daily=None,
        start_year=int(raw_config.get("start_year", 1994)),
        end_year=int(raw_config.get("end_year", 2026)),
    )
    catalog = _build_weekly_spec_catalog(examples)
    features = sorted({spec.split("|", 1)[0] for spec in catalog})
    spec_counts = Counter(spec.split("|", 1)[0] for spec in catalog)
    rows = [
        {
            "feature": feature,
            "spec_count": int(spec_counts[feature]),
            "family": _feature_family(feature),
        }
        for feature in features
    ]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / f"{args.file_prefix}_feature_catalog.csv", index=False)
    family_counts = Counter(row["family"] for row in rows)
    summary = {
        "examples": int(len(examples)),
        "feature_count": int(len(features)),
        "catalog_specs": int(len(catalog)),
        "available_assets": _available_assets_from_examples(examples),
        "tradable_assets": list(TRADABLE_ASSETS),
        "family_counts": dict(sorted(family_counts.items())),
    }
    (output / f"{args.file_prefix}_feature_catalog_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _feature_family(feature: str) -> str:
    assets = tuple(asset.lower() for asset in TRADABLE_ASSETS)
    if feature.startswith("asset_"):
        return "asset_weekly_momentum_volatility"
    if feature.startswith("calendar_") or feature in {
        "election_year_dummy",
        "house_control_party",
        "is_election_year",
        "is_post_election_year",
        "unified_government",
    }:
        return "calendar_political_cycle"
    if feature.startswith(("vix", "vvix", "skew")):
        return "volatility_vix_complex"
    if feature.startswith(("tnx", "irx", "fvx", "tyx")):
        return "rates_yields"
    if feature.startswith("spy_vs_") or feature.endswith("_spy_level") or "_spy_ret_" in feature:
        return "relative_strength_ratios"
    if feature.startswith(("xlk_xlu", "xly_xlp")):
        return "relative_strength_ratios"
    if any(feature.startswith(f"{asset}_") for asset in assets):
        return "tradable_asset_returns_volatility_ratios"
    if feature.startswith("free_epu"):
        return "economic_policy_uncertainty"
    if feature.startswith(("vol_", "atr_")) or feature == "volatility_spike_dummy":
        return "realized_volatility_range_estimators"
    if feature.startswith(("gap_", "intraday", "close_location", "return_intraday")):
        return "intraday_gap_close_location"
    if feature.startswith("dollar_volume"):
        return "volume_liquidity_proxy"
    return "other"


if __name__ == "__main__":
    raise SystemExit(main())
