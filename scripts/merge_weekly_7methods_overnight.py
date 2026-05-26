from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.config import load_yaml  # noqa: E402
from trading_lab.data_loader import load_market_data  # noqa: E402
from trading_lab.public_data import download_yahoo_chart  # noqa: E402
from trading_lab.weekly_7methods_stateful import merge_stateful_weekly_leaderboards  # noqa: E402
from trading_lab.weekly_multi_asset import build_weekly_multi_asset_examples  # noqa: E402


STAGE_FILE_RE = re.compile(r"weekly_7methods_overnight_leaderboard_stage_(?P<method>.+)_(?P<wave>\d+)_(?P<stage>\d+)\.csv$")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge available overnight weekly 7-method artifacts.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", default="outputs/weekly_7methods_overnight")
    parser.add_argument("--config", default="configs/weekly_7methods_overnight.yaml")
    parser.add_argument("--max-output-rows", type=int, default=50_000)
    parser.add_argument("--file-prefix", default="weekly_7methods_overnight")
    parser.add_argument("--expected-waves", type=int, default=None)
    parser.add_argument("--expected-stages-per-method", type=int, default=None)
    args = parser.parse_args()

    raw_config = load_yaml(args.config)
    paths = sorted(glob.glob(args.input_glob, recursive=True))
    methods = list(raw_config.get("methods") or [])
    expected_waves = int(args.expected_waves or raw_config.get("waves", 10))
    expected_stages = int(args.expected_stages_per_method or raw_config.get("stages_per_method", 35))

    examples = _load_examples(raw_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = merge_stateful_weekly_leaderboards(
        paths,
        output_dir,
        examples=examples,
        max_output_rows=args.max_output_rows,
        file_prefix=args.file_prefix,
        expected_methods=methods,
    )
    partial_summary = _partial_summary(paths, methods=methods, expected_waves=expected_waves, expected_stages=expected_stages)
    final_summary = {
        **summary,
        **partial_summary,
        "partial": bool(partial_summary["artifacts_downloaded"] < partial_summary["expected_artifacts"]),
        "locked_opened": False,
        "validation_role": "report_only",
    }
    summary_path = output_dir / f"{args.file_prefix}_summary.json"
    summary_path.write_text(json.dumps(final_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(final_summary, indent=2, sort_keys=True))
    return 0


def _load_examples(raw_config: dict[str, object]):
    data_path = Path(str(raw_config.get("data_path", "data/public/spy_daily.csv")))
    if data_path.exists():
        daily = load_market_data(data_path)
    else:
        daily = download_yahoo_chart("SPY")
    try:
        benchmark_daily = download_yahoo_chart("^GSPC")
    except Exception:
        benchmark_daily = None
    return build_weekly_multi_asset_examples(
        daily,
        benchmark_daily=benchmark_daily,
        start_year=int(raw_config.get("start_year", 1994)),
        end_year=int(raw_config.get("end_year", 2026)),
    )


def _partial_summary(paths: list[str], *, methods: list[str], expected_waves: int, expected_stages: int) -> dict[str, object]:
    waves_found: set[int] = set()
    files_by_method: dict[str, int] = {method: 0 for method in methods}
    stages_by_method: dict[str, set[str]] = {method: set() for method in methods}
    stages_by_wave_method: dict[str, dict[str, int]] = defaultdict(lambda: {method: 0 for method in methods})
    matched = 0
    for raw_path in paths:
        match = STAGE_FILE_RE.search(Path(raw_path).name)
        if not match:
            continue
        matched += 1
        method = match.group("method")
        wave = int(match.group("wave"))
        stage = int(match.group("stage"))
        waves_found.add(wave)
        files_by_method[method] = int(files_by_method.get(method, 0)) + 1
        stages_by_method.setdefault(method, set()).add(f"{wave}:{stage}")
        stages_by_wave_method[str(wave)][method] = int(stages_by_wave_method[str(wave)].get(method, 0)) + 1

    expected_artifacts = max(0, expected_waves) * max(0, expected_stages) * max(1, len(methods))
    return {
        "artifacts_downloaded": int(matched),
        "expected_artifacts": int(expected_artifacts),
        "expected_waves": int(expected_waves),
        "expected_stages_per_method": int(expected_stages),
        "waves_found": sorted(waves_found),
        "wave_count_found": int(len(waves_found)),
        "stage_files_by_method": files_by_method,
        "unique_wave_stage_by_method": {method: int(len(stages_by_method.get(method, set()))) for method in methods},
        "stage_files_by_wave_method": dict(sorted(stages_by_wave_method.items(), key=lambda item: int(item[0]))),
    }


if __name__ == "__main__":
    raise SystemExit(main())
