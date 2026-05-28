from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading_lab.config import load_yaml  # noqa: E402
from trading_lab.data_loader import load_market_data  # noqa: E402
from trading_lab.weekly_multi_asset import (  # noqa: E402
    WEEKLY_MAX_SHARPE_SCORE_MODE,
    _candidate_from_row,
    _ml_candidate_from_row,
    build_weekly_multi_asset_examples,
    evaluate_weekly_machine_learning_candidate,
    evaluate_weekly_multi_asset_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run universal robustness on strategy returns.")
    parser.add_argument("--input", required=True, help="CSV with returns or leaderboard rows")
    parser.add_argument(
        "--input-kind",
        choices=("returns-csv", "weekly-positions", "weekly-leaderboard"),
        default="returns-csv",
    )
    parser.add_argument("--config", default="configs/weekly_spy_sharpe_4methods_180.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aurora-root", default=str(Path.home() / "QuantForge"))
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--returns-are-net", default="true")
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    args = parser.parse_args()

    _prepare_aurora_import(Path(args.aurora_root))
    from aurora.validation.robustness_config import UniversalRobustnessConfig
    from aurora.validation.robustness_reports import write_universal_robustness_outputs
    from aurora.validation.universal_robustness import run_batch_universal_robustness

    if args.input_kind == "returns-csv":
        returns = pd.read_csv(args.input)
    elif args.input_kind == "weekly-positions":
        positions = pd.read_csv(args.input)
        method = str(positions.get("method", pd.Series(["manual"])).iloc[0])
        returns = _weekly_positions_to_universal_returns(
            positions,
            str(args.run_id),
            method,
            int(args.n_trials or 1),
            _truthy(args.returns_are_net),
        )
    else:
        returns = weekly_leaderboard_to_universal_returns(
            Path(args.input),
            config_path=Path(args.config),
            run_id=str(args.run_id),
            n_trials=args.n_trials,
            returns_are_net=_truthy(args.returns_are_net),
        )

    config_kwargs: dict[str, Any] = {}
    if args.bootstrap_samples is not None:
        config_kwargs["bootstrap_samples"] = int(args.bootstrap_samples)
    if args.bootstrap_seed is not None:
        config_kwargs["bootstrap_seed"] = int(args.bootstrap_seed)
    config = UniversalRobustnessConfig(**config_kwargs)
    outputs = run_batch_universal_robustness(returns, config)
    unsupported = returns.attrs.get("unsupported_rows")
    if unsupported is not None and not unsupported.empty:
        outputs["results"] = pd.concat([outputs["results"], unsupported], ignore_index=True)
        outputs["methods"] = _method_summary(outputs["results"])
        unsupported_failures = unsupported[["candidate_id", "fail_reasons"]].rename(columns={"fail_reasons": "fail_reason"})
        outputs["fail_reasons"] = pd.concat([outputs["fail_reasons"], unsupported_failures], ignore_index=True)
    write_universal_robustness_outputs(
        args.output_dir,
        results=outputs["results"],
        pass_results=outputs["pass"],
        methods=outputs["methods"],
        year_by_year=outputs["year_by_year"],
        fail_reasons=outputs["fail_reasons"],
        duplicates=outputs["duplicates"],
        bootstrap=outputs["bootstrap"],
        data_quality=outputs["data_quality"],
        config=config,
    )
    return 0


def weekly_leaderboard_to_universal_returns(
    leaderboard_path: Path,
    *,
    config_path: Path,
    run_id: str,
    n_trials: int | None,
    returns_are_net: bool,
) -> pd.DataFrame:
    leaderboard = pd.read_csv(leaderboard_path)
    raw_config = load_yaml(config_path)
    daily = load_market_data(raw_config.get("data_path", "data/public/spy_daily.csv"))
    examples = build_weekly_multi_asset_examples(
        daily,
        benchmark_daily=None,
        start_year=int(raw_config.get("start_year", 1994)),
        end_year=int(raw_config.get("end_year", 2026)),
    )
    rows: list[pd.DataFrame] = []
    unsupported_rows: list[dict[str, Any]] = []
    inferred_trials = int(n_trials or len(leaderboard))
    for _, row in leaderboard.iterrows():
        method = str(row.get("method", "") or "")
        candidate_id = str(row.get("candidate_id", "") or "")
        if method == "aurora_ml":
            unsupported_rows.append(_unsupported_row(candidate_id, run_id, method))
            continue
        try:
            if method in {"github_ml", "machine_learning"}:
                candidate = _ml_candidate_from_row(row)
                _, positions, _ = evaluate_weekly_machine_learning_candidate(
                    examples,
                    candidate,
                    method=method,
                    score_mode=str(raw_config.get("score_mode", WEEKLY_MAX_SHARPE_SCORE_MODE)),
                )
            else:
                candidate = _candidate_from_row(row)
                _, positions, _ = evaluate_weekly_multi_asset_candidate(
                    examples,
                    candidate,
                    method=method,
                    score_mode=str(raw_config.get("score_mode", WEEKLY_MAX_SHARPE_SCORE_MODE)),
                )
        except Exception as exc:  # pragma: no cover - defensive for malformed external CSVs
            unsupported_rows.append(_unsupported_row(candidate_id, run_id, method, f"reconstruction_failed:{exc}"))
            continue
        if candidate_id:
            positions["candidate_id"] = candidate_id
        rows.append(_weekly_positions_to_universal_returns(positions, run_id, method, inferred_trials, returns_are_net))
    returns = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=_UNIVERSAL_COLUMNS)
    if unsupported_rows:
        returns.attrs["unsupported_rows"] = pd.DataFrame(unsupported_rows)
    return returns


def _weekly_positions_to_universal_returns(
    positions: pd.DataFrame,
    run_id: str,
    method: str,
    n_trials: int,
    returns_are_net: bool,
) -> pd.DataFrame:
    role_map = {"train": "is", "validation": "oos", "locked": "locked"}
    out = pd.DataFrame(
        {
            "candidate_id": positions["candidate_id"].astype(str),
            "run_id": run_id,
            "source_method": method,
            "timestamp": pd.to_datetime(positions["target_week_end"]),
            "frequency": "weekly",
            "strategy_return": pd.to_numeric(positions["strategy_return"], errors="coerce"),
            "returns_are_net": bool(returns_are_net),
            "sample_role": positions["period"].map(role_map).fillna(positions["period"]),
            "n_trials": int(n_trials),
            "benchmark_return": pd.to_numeric(positions.get("spy_return"), errors="coerce"),
            "exposure": pd.to_numeric(positions.get("exposure"), errors="coerce"),
            "turnover": pd.to_numeric(positions.get("exposure"), errors="coerce").diff().abs().fillna(0.0),
            "traded_asset": positions.get("traded_asset", ""),
            "target_year": positions.get("target_year", pd.NA),
        }
    )
    return out


def _unsupported_row(
    candidate_id: str,
    run_id: str,
    method: str,
    reason: str = "unsupported_missing_returns",
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "run_id": run_id,
        "source_method": method,
        "unsupported_reason": reason,
        "data_quality_pass": False,
        "statistical_pass": False,
        "multiple_testing_pass": False,
        "cost_pass": False,
        "robust_pass": False,
        "portfolio_eligible": False,
        "fail_reasons": reason,
    }


def _prepare_aurora_import(aurora_root: Path) -> None:
    init_file = aurora_root / "__init__.py"
    if not init_file.exists():
        return
    for name in list(sys.modules):
        if name == "aurora" or name.startswith("aurora."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "aurora",
        init_file,
        submodule_search_locations=[str(aurora_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load aurora package from {aurora_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["aurora"] = module
    spec.loader.exec_module(module)


def _method_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "source_method" not in results.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for method, group in results.groupby("source_method", dropna=False):
        rows.append(
            {
                "source_method": str(method),
                "candidates": int(len(group)),
                "robust_pass": int(group.get("robust_pass", pd.Series(dtype=bool)).fillna(False).sum()),
                "portfolio_eligible": int(
                    group.get("portfolio_eligible", pd.Series(dtype=bool)).fillna(False).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


_UNIVERSAL_COLUMNS = (
    "candidate_id",
    "run_id",
    "source_method",
    "timestamp",
    "frequency",
    "strategy_return",
    "returns_are_net",
    "sample_role",
    "n_trials",
)


if __name__ == "__main__":
    raise SystemExit(main())
