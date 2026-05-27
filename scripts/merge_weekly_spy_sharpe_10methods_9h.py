from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.merge_weekly_spy_sharpe_4methods_180 import merge_outputs  # noqa: E402
from trading_lab.config import load_yaml  # noqa: E402
from trading_lab.monthly_risk import _json_safe  # noqa: E402


DEFAULT_METHODS = (
    "beam",
    "genetic",
    "sobol_random_asha_real",
    "optuna_tpe_hyperband",
    "dehb_real",
    "bohb_real",
    "smac_mf_real",
    "bandit",
    "aurora_ml",
    "github_ml",
)

STAGE_FILE_RE = re.compile(r"weekly_spy_sharpe_10methods_9h_leaderboard_stage_(?P<method>.+)_(?P<wave>\d+)_(?P<stage>\d+)\.csv$")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge weekly SPY Sharpe 10-method 9h wave outputs.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", default="outputs/weekly_spy_sharpe_10methods_9h")
    parser.add_argument("--config", default="configs/weekly_spy_sharpe_10methods_9h_waves.yaml")
    parser.add_argument("--file-prefix", default="weekly_spy_sharpe_10methods_9h")
    parser.add_argument("--expected-jobs", type=int, default=900)
    args = parser.parse_args()

    raw_config = load_yaml(args.config)
    methods = [str(item) for item in raw_config.get("methods", DEFAULT_METHODS)]
    summary = merge_outputs(
        input_glob=args.input_glob,
        output_dir=args.output_dir,
        file_prefix=args.file_prefix,
        expected_jobs=args.expected_jobs,
        expected_methods=methods,
    )
    partial = _partial_summary(args.input_glob, methods=methods, expected_waves=int(raw_config.get("waves", 5)), expected_stages=int(raw_config.get("stages_per_method", 18)))
    final_summary = {
        **summary,
        **partial,
        "artifact": "weekly-spy-sharpe-10methods-9h-waves-leaderboard",
        "methods": methods,
        "partial": bool(partial["artifacts_downloaded"] < partial["expected_artifacts"]),
        "locked_opened": False,
        "validation_role": "report_only",
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.file_prefix}_summary.json").write_text(json.dumps(_json_safe(final_summary), indent=2, sort_keys=True), encoding="utf-8")
    if not (output / f"{args.file_prefix}_state_summary.json").exists():
        (output / f"{args.file_prefix}_state_summary.json").write_text(
            json.dumps(
                {
                    "methods": methods,
                    "waves_found": partial["waves_found"],
                    "stateful": True,
                    "locked_opened": False,
                    "validation_role": "report_only",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    print(json.dumps(final_summary, indent=2, sort_keys=True))
    return 0


def _partial_summary(input_glob: str, *, methods: list[str], expected_waves: int, expected_stages: int) -> dict[str, Any]:
    import glob

    paths = sorted(glob.glob(input_glob, recursive=True))
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
