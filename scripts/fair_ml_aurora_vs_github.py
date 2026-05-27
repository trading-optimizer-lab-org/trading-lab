from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trading_lab.config import load_yaml  # noqa: E402
from trading_lab.data_loader import load_market_data  # noqa: E402
from trading_lab.public_data import download_public_data  # noqa: E402
from trading_lab.weekly_multi_asset import build_weekly_multi_asset_examples  # noqa: E402


VALID_RULE = {
    "train_calmar_gt": 0.0,
    "validation_calmar_gt": 0.0,
    "validation_calmar_ratio_min": None,
    "train_cagr_min": None,
    "validation_cagr_min": None,
    "locked_opened": False,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fair GitHub benchmark for Aurora ML vs trading-lab ML.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run-pair")
    run.add_argument("--track", required=True, choices=("normalized", "native"))
    run.add_argument("--order", required=True, choices=("aurora_first", "github_first"))
    run.add_argument("--seconds-per-engine", type=float, default=480.0)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--quantforge-dir", default="")
    run.add_argument("--trading-lab-dir", default=str(ROOT))
    run.add_argument("--config", default="configs/weekly_sharpe_3methods_max_parallel_900.yaml")
    run.add_argument("--seed", type=int, default=20260527)
    run.add_argument("--aurora-models", default="logistic,forest,ridge,corr")

    merge = sub.add_parser("merge")
    merge.add_argument("--input-root", required=True)
    merge.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    if args.cmd == "run-pair":
        run_pair(args)
    elif args.cmd == "merge":
        merge_results(args)
    return 0


def run_pair(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    trading_lab_dir = Path(args.trading_lab_dir).resolve()
    config_path = trading_lab_dir / args.config
    base_config = load_yaml(config_path)
    data_path = trading_lab_dir / base_config.get("data_path", "data/public/spy_daily.csv")
    if not data_path.exists():
        download_public_data(data_path)
    daily = load_market_data(data_path)

    if args.track == "normalized":
        run_data = _normalized_daily(daily)
        run_config = dict(base_config)
        normalized_csv = work_dir / "normalized_spy_daily.csv"
        _write_daily_csv(run_data, normalized_csv)
        run_config["data_path"] = str(normalized_csv)
        run_config_path = work_dir / "normalized_config.yaml"
        run_config_path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
        github_config = run_config_path
        aurora_models = str(args.aurora_models)
        github_total_stages = 1
    else:
        run_data = daily
        github_config = config_path
        aurora_models = str(args.aurora_models)
        github_total_stages = 1

    engines = ["aurora", "github_ml"] if args.order == "aurora_first" else ["github_ml", "aurora"]
    results: list[dict[str, Any]] = []
    raw_dirs: dict[str, str] = {}
    for index, engine in enumerate(engines, start=1):
        engine_dir = output_dir / f"{index:02d}_{engine}"
        engine_dir.mkdir(parents=True, exist_ok=True)
        if engine == "aurora":
            result = _run_aurora(
                trading_lab_dir=trading_lab_dir,
                daily=run_data,
                output_dir=engine_dir,
                seconds=float(args.seconds_per_engine),
                models=aurora_models,
                seed=int(args.seed) + index,
            )
        else:
            result = _run_github_ml(
                trading_lab_dir=trading_lab_dir,
                config_path=github_config,
                output_dir=engine_dir,
                seconds=float(args.seconds_per_engine),
                seed=int(args.seed) + index,
                total_stages=github_total_stages,
            )
        result.update(
            {
                "track": args.track,
                "order": args.order,
                "run_position": index,
                "engine": engine,
            }
        )
        raw_dirs[engine] = str(engine_dir)
        results.append(result)

    (output_dir / "fair_ml_pair_summary.json").write_text(
        json.dumps(
            {
                "track": args.track,
                "order": args.order,
                "seconds_per_engine": float(args.seconds_per_engine),
                "valid_rule": VALID_RULE,
                "environment": _environment(),
                "raw_dirs": raw_dirs,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_csv(output_dir / "fair_ml_pair_results.csv", results)


def _run_aurora(
    *,
    trading_lab_dir: Path,
    daily: pd.DataFrame,
    output_dir: Path,
    seconds: float,
    models: str,
    seed: int,
) -> dict[str, Any]:
    qf_data = output_dir / "qf_data"
    qf_run_root = output_dir / "runs"
    qf_data.mkdir(parents=True, exist_ok=True)
    qf_run_root.mkdir(parents=True, exist_ok=True)
    store_script = output_dir / "load_aurora_store.py"
    daily_csv = output_dir / "aurora_input.csv"
    _write_daily_csv(daily, daily_csv)
    store_script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import json",
                "from multiprocessing import freeze_support",
                "import pandas as pd",
                "from aurora.data_contracts.timeseries_store import TimeSeriesStore",
                "from aurora.research.ml_search import MLSearchConfig, run_ml_search",
                "",
                "def main():",
                f"    df = pd.read_csv(r'{daily_csv}', parse_dates=['timestamp']).set_index('timestamp')",
                "    store = TimeSeriesStore()",
                "    store.put('prices_daily', 'SPY', df, version='fair_ml_spy', replace=True)",
                "    report = run_ml_search(",
                "        MLSearchConfig(",
                "            run_id='fair_ml_aurora',",
                "            symbol='SPY',",
                "            library='prices_daily',",
                "            target_calmar=1,",
                "            validation_target_calmar=1,",
                "            workers=6,",
                "            max_candidates=1000000,",
                "            batch_size=12,",
                f"            seed={int(seed)},",
                f"            time_limit_seconds={max(1.0, seconds)!r},",
                f"            models={tuple(model.strip() for model in models.split(',') if model.strip())!r},",
                f"            run_root=r'{qf_run_root}',",
                "            top_n=100,",
                "            target_objective_count=999999,",
                "            min_train_cagr=0.04,",
                "            min_validation_cagr=0.03,",
                "            max_train_validation_calmar_ratio=1.25,",
                "            defer_robustness_until_basic_pass=True,",
                "            no_costs=True,",
                "            no_locked=True,",
                "        )",
                "    )",
                "    print(json.dumps(report.to_dict(), default=str))",
                "",
                "if __name__ == '__main__':",
                "    freeze_support()",
                "    main()",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(trading_lab_dir / "src")
    env["QF_DATA_DIR"] = str(qf_data)
    started = time.perf_counter()
    proc = _run_command([sys.executable, str(store_script)], cwd=trading_lab_dir, env=env, log=output_dir / "aurora_ml.log", check=False)
    wall_seconds = time.perf_counter() - started
    ml_dir = qf_run_root / "fair_ml_aurora" / "ml_search"
    candidates = _read_aurora_candidates(ml_dir / "candidates.jsonl")
    status = _read_json(ml_dir / "status.json")
    return _summarize_candidates(
        engine="aurora",
        candidates=candidates,
        reported_evaluated=int(status.get("candidates_evaluated", len(candidates)) or len(candidates)),
        effective_seconds=float(status.get("elapsed_seconds", wall_seconds) or wall_seconds),
        wall_seconds=wall_seconds,
        exit_code=proc.returncode,
        engine_failed=not bool(status),
        objective_met=bool(status.get("objective_met", False)),
        locked_opened=bool(status.get("locked_opened", False)),
        raw_output_dir=str(ml_dir),
    )


def _run_github_ml(
    *,
    trading_lab_dir: Path,
    config_path: Path,
    output_dir: Path,
    seconds: float,
    seed: int,
    total_stages: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/run_weekly_7methods_12h_stateful_search.py",
        "--config",
        str(config_path),
        "--method",
        "machine_learning",
        "--wave",
        "1",
        "--stage",
        "0",
        "--total-stages",
        str(total_stages),
        "--time-budget-minutes",
        str(max(1.0, seconds) / 60.0),
        "--seed-pool",
        "200000",
        "--beam-width",
        "320",
        "--generations",
        "80",
        "--mutations-per-parent",
        "80",
        "--max-features",
        "12",
        "--top-rows-per-stage",
        "50000",
        "--output-dir",
        str(output_dir),
        "--file-prefix",
        "fair_ml_github",
        "--random-seed",
        str(seed),
    ]
    started = time.perf_counter()
    proc = _run_command(cmd, cwd=trading_lab_dir, env=os.environ.copy(), log=output_dir / "github_ml.log", check=False)
    wall_seconds = time.perf_counter() - started
    rows = _read_csv_dicts(output_dir / "fair_ml_github_leaderboard.csv")
    max_elapsed = max([_to_float(row.get("elapsed_seconds"), 0.0) for row in rows] or [wall_seconds])
    locked_opened = any(_to_bool(row.get("locked_opened")) for row in rows)
    return _summarize_candidates(
        engine="github_ml",
        candidates=rows,
        reported_evaluated=len(rows),
        effective_seconds=max_elapsed,
        wall_seconds=wall_seconds,
        exit_code=proc.returncode,
        engine_failed=proc.returncode != 0,
        objective_met=bool(rows),
        locked_opened=locked_opened,
        raw_output_dir=str(output_dir),
    )


def merge_results(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_files = sorted(input_root.rglob("fair_ml_pair_summary.json"))
    pair_summaries = [_read_json(path) for path in pair_files]
    rows: list[dict[str, Any]] = []
    valid_candidates: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    for summary in pair_summaries:
        for row in summary.get("results", []):
            rows.append(row)
            timing.append(
                {
                    "track": row.get("track"),
                    "order": row.get("order"),
                    "engine": row.get("engine"),
                    "run_position": row.get("run_position"),
                    "effective_seconds": row.get("effective_seconds"),
                    "wall_seconds": row.get("wall_seconds"),
                    "exit_code": row.get("exit_code"),
                    "engine_failed": row.get("engine_failed"),
                    "objective_met": row.get("objective_met"),
                    "raw_output_dir": row.get("raw_output_dir"),
                }
            )
            best = row.get("best_valid") or {}
            if best:
                valid_candidates.append({"track": row.get("track"), "engine": row.get("engine"), **best})

    normalized = [row for row in rows if row.get("track") == "normalized"]
    native = [row for row in rows if row.get("track") == "native"]
    summary_rows = _aggregate(rows)
    official = _winner(_aggregate(normalized))
    final_summary = {
        "artifact": "fair-ml-aurora-vs-github-30m-results",
        "valid_rule": VALID_RULE,
        "pairs_found": len(pair_summaries),
        "official_track": "normalized",
        "official_winner": official,
        "partial": len(pair_summaries) < 4,
        "environment": pair_summaries[0].get("environment", _environment()) if pair_summaries else _environment(),
        "summary": summary_rows,
    }
    _write_csv(output_dir / "fair_ml_results.csv", rows)
    _write_csv(output_dir / "fair_ml_normalized_results.csv", normalized)
    _write_csv(output_dir / "fair_ml_native_results.csv", native)
    _write_csv(output_dir / "fair_ml_valid_candidates.csv", valid_candidates)
    _write_csv(output_dir / "fair_ml_engine_timing.csv", timing)
    _write_csv(output_dir / "fair_ml_summary_table.csv", summary_rows)
    (output_dir / "fair_ml_summary.json").write_text(json.dumps(final_summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "fair_ml_environment.json").write_text(
        json.dumps(final_summary["environment"], indent=2, sort_keys=True), encoding="utf-8"
    )


def _summarize_candidates(
    *,
    engine: str,
    candidates: list[dict[str, Any]],
    reported_evaluated: int,
    effective_seconds: float,
    wall_seconds: float,
    exit_code: int,
    engine_failed: bool,
    objective_met: bool,
    locked_opened: bool,
    raw_output_dir: str,
) -> dict[str, Any]:
    valid = [row for row in candidates if _valid_candidate(row, locked_opened=locked_opened)]
    best_valid = max(valid, key=lambda row: (_metric(row, "validation_calmar"), _metric(row, "train_calmar")), default=None)
    first_valid_seconds = min([_to_float(row.get("elapsed_seconds"), 0.0) for row in valid] or [float("nan")])
    minutes = max(effective_seconds / 60.0, 1e-9)
    return {
        "engine": engine,
        "evaluated": int(reported_evaluated),
        "rows_available": len(candidates),
        "valid": len(valid),
        "effective_seconds": float(effective_seconds),
        "wall_seconds": float(wall_seconds),
        "evaluated_per_min": float(reported_evaluated / minutes),
        "valid_per_min": float(len(valid) / minutes),
        "first_valid_seconds": first_valid_seconds,
        "best_valid": _candidate_summary(best_valid),
        "best_train_calmar": _best_metric(candidates, "train_calmar"),
        "best_validation_calmar": _best_metric(candidates, "validation_calmar"),
        "best_validation_cagr": _best_metric(candidates, "validation_cagr"),
        "locked_opened": bool(locked_opened),
        "exit_code": int(exit_code),
        "engine_failed": bool(engine_failed),
        "objective_met": bool(objective_met),
        "raw_output_dir": raw_output_dir,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("track")), str(row.get("engine"))), []).append(row)
    out = []
    for (track, engine), group in sorted(grouped.items()):
        effective = sum(_to_float(row.get("effective_seconds"), 0.0) for row in group)
        evaluated = sum(int(row.get("evaluated", 0) or 0) for row in group)
        valid = sum(int(row.get("valid", 0) or 0) for row in group)
        best_valids = [row.get("best_valid") for row in group if row.get("best_valid")]
        best = max(best_valids, key=lambda row: (_to_float(row.get("validation_calmar"), -1e9), _to_float(row.get("train_calmar"), -1e9)), default=None)
        minutes = max(effective / 60.0, 1e-9)
        out.append(
            {
                "track": track,
                "engine": engine,
                "repetitions": len(group),
                "effective_seconds": effective,
                "evaluated": evaluated,
                "valid": valid,
                "evaluated_per_min": evaluated / minutes,
                "valid_per_min": valid / minutes,
                "best_valid_candidate": "" if best is None else best.get("candidate_id", ""),
                "best_validation_calmar": "" if best is None else best.get("validation_calmar", ""),
                "best_train_calmar": "" if best is None else best.get("train_calmar", ""),
            }
        )
    return out


def _winner(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            _to_float(row.get("valid_per_min"), 0.0),
            _to_float(row.get("best_validation_calmar"), -1e9),
            _to_float(row.get("best_train_calmar"), -1e9),
            _to_float(row.get("evaluated_per_min"), 0.0),
        ),
    )


def _valid_candidate(row: dict[str, Any], *, locked_opened: bool) -> bool:
    train_calmar = _metric(row, "train_calmar")
    validation_calmar = _metric(row, "validation_calmar")
    train_cagr = _metric(row, "train_cagr")
    validation_cagr = _metric(row, "validation_cagr")
    ratio_min = VALID_RULE.get("validation_calmar_ratio_min")
    train_cagr_min = VALID_RULE.get("train_cagr_min")
    validation_cagr_min = VALID_RULE.get("validation_cagr_min")
    if locked_opened or _to_bool(row.get("locked_opened")):
        return False
    return (
        train_calmar > float(VALID_RULE["train_calmar_gt"])
        and validation_calmar > float(VALID_RULE["validation_calmar_gt"])
        and (ratio_min is None or float(ratio_min) <= 0 or validation_calmar >= float(ratio_min) * train_calmar)
        and (train_cagr_min is None or train_cagr >= float(train_cagr_min))
        and (validation_cagr_min is None or validation_cagr >= float(validation_cagr_min))
    )


def _candidate_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    train_calmar = _metric(row, "train_calmar")
    validation_calmar = _metric(row, "validation_calmar")
    return {
        "candidate_id": row.get("candidate_id", ""),
        "model": row.get("model") or row.get("ml_model") or "",
        "train_calmar": train_calmar,
        "validation_calmar": validation_calmar,
        "validation_train_ratio": validation_calmar / train_calmar if train_calmar > 0 else None,
        "train_cagr": _metric(row, "train_cagr"),
        "validation_cagr": _metric(row, "validation_cagr"),
    }


def _metric(row: dict[str, Any], name: str) -> float:
    if name in row:
        return _to_float(row.get(name), float("nan"))
    if name.startswith("train_"):
        return _to_float((row.get("train_metrics") or {}).get(name.removeprefix("train_")), float("nan"))
    if name.startswith("validation_"):
        return _to_float((row.get("validation_metrics") or {}).get(name.removeprefix("validation_")), float("nan"))
    return float("nan")


def _best_metric(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [_metric(row, name) for row in rows]
    values = [value for value in values if pd.notna(value)]
    return max(values) if values else None


def _read_aurora_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _normalized_daily(daily: pd.DataFrame) -> pd.DataFrame:
    cols = [column for column in ("open", "high", "low", "close", "volume") if column in daily]
    out = daily.loc[:, cols].copy()
    out["adj_close"] = out["close"]
    return out


def _write_daily_csv(daily: pd.DataFrame, path: Path) -> None:
    out = daily.copy()
    out = out.reset_index().rename(columns={out.index.name or "index": "timestamp"})
    if "timestamp" not in out:
        out = out.rename(columns={out.columns[0]: "timestamp"})
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True)
    log.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed with {proc.returncode}: {' '.join(cmd)}")
    return proc


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cwd": str(Path.cwd()),
        "github_actions": os.environ.get("GITHUB_ACTIONS", "false"),
        "runner_name": os.environ.get("RUNNER_NAME", ""),
        "runner_os": os.environ.get("RUNNER_OS", ""),
        "runner_arch": os.environ.get("RUNNER_ARCH", ""),
        "git_sha": os.environ.get("GITHUB_SHA", ""),
    }


if __name__ == "__main__":
    raise SystemExit(main())
