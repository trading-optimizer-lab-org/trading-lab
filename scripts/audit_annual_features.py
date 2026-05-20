from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.annual_prediction import audit_annual_feature_coverage, build_annual_examples  # noqa: E402
from trading_lab.config import load_yaml  # noqa: E402
from trading_lab.data_loader import load_market_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit annual feature coverage and quality.")
    parser.add_argument("--config", default="configs/annual_sp500_beam.yaml")
    parser.add_argument("--output", default="outputs/annual_sp500_beam/annual_feature_coverage.csv")
    parser.add_argument("--summary", default="outputs/annual_sp500_beam/annual_feature_coverage_summary.json")
    args = parser.parse_args()

    raw_config = load_yaml(args.config)
    daily = load_market_data(raw_config.get("data_path", "data/public/sp500_annual_daily.csv"))
    examples = build_annual_examples(
        daily,
        start_year=int(raw_config.get("start_year", 1980)),
        end_year=int(raw_config.get("end_year", 2025)),
    )
    audit = audit_annual_feature_coverage(examples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)
    summary = {
        "features_total": int(len(audit)),
        "usable_in_beam": int(audit["usable_in_beam"].sum()),
        "quality_counts": {str(key): int(value) for key, value in audit["quality"].value_counts().to_dict().items()},
        "output": str(output),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
