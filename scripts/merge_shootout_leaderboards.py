from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.results import save_leaderboard


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge equal-budget SPY-only shootout results.")
    parser.add_argument("--input-glob", default="outputs/survival_spy_only/survival_spy_only_*_stage_*.csv")
    parser.add_argument("--output", default="outputs/survival_spy_only/shootout_leaderboard.csv")
    parser.add_argument("--summary", default="outputs/survival_spy_only/shootout_summary.json")
    parser.add_argument("--method-table", default="outputs/survival_spy_only/shootout_methods.csv")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    leaderboard = _read_inputs(paths)
    if not leaderboard.empty:
        leaderboard["shootout_method"] = leaderboard.apply(_method_for_row, axis=1)
        leaderboard["quick_pass"] = leaderboard.apply(_quick_pass, axis=1)
        leaderboard["near_final"] = leaderboard.apply(_near_final, axis=1)
        leaderboard["shootout_score"] = leaderboard.apply(_shootout_score, axis=1)
        leaderboard = leaderboard.sort_values("shootout_score", ascending=False)
    output_path = save_leaderboard(leaderboard, args.output)
    method_table = _method_table(leaderboard)
    Path(args.method_table).parent.mkdir(parents=True, exist_ok=True)
    method_table.to_csv(args.method_table, index=False)
    summary = {
        "input_files": paths,
        "rows": int(len(leaderboard)),
        "candidates_evaluated": int(len(leaderboard)),
        "accepted_strict": int(leaderboard["accepted"].sum()) if "accepted" in leaderboard else 0,
        "quick_pass": int(leaderboard["quick_pass"].sum()) if "quick_pass" in leaderboard else 0,
        "near_final": int(leaderboard["near_final"].sum()) if "near_final" in leaderboard else 0,
        "method_table": method_table.to_dict(orient="records"),
        "best": {} if leaderboard.empty else leaderboard.iloc[0].to_dict(),
        "output": str(output_path),
        "locked_opened": False,
    }
    clean_summary = _json_clean(summary)
    Path(args.summary).write_text(json.dumps(clean_summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(clean_summary, indent=2, default=str))
    return 0


def _read_inputs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["source_file"] = Path(path).name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _method_for_row(row: pd.Series) -> str:
    if isinstance(row.get("marathon_method"), str) and row["marathon_method"] not in {"", "nan"}:
        return str(row["marathon_method"])
    if isinstance(row.get("meta_method"), str) and row["meta_method"] not in {"", "nan"}:
        return str(row["meta_method"])
    rule = str(row.get("rule", ""))
    candidate = str(row.get("candidate_id", ""))
    if "beam_generation" in row and pd.notna(row.get("beam_generation")):
        return "beam"
    if "adaptive_stage" in candidate:
        return "adaptive"
    if rule in {"spy_long_short_always", "spy_long_short_score"}:
        source = str(row.get("source_file", ""))
        for method in ("adaptive", "beam", "bayesian", "bandit", "genetic"):
            if method in source or method in candidate:
                return method
    return "adaptive"


def _quick_pass(row: pd.Series) -> bool:
    return (
        float(row.get("train_calmar", 0.0) or 0.0) >= 0.50
        and float(row.get("validation_calmar", 0.0) or 0.0) >= 0.50
        and abs(float(row.get("train_mdd", 0.0) or 0.0)) <= 0.45
        and abs(float(row.get("validation_mdd", 0.0) or 0.0)) <= 0.45
        and 6.0 <= float(row.get("trades_per_year", 0.0) or 0.0) <= 150.0
        and 0.20 <= float(row.get("long_fraction", 0.0) or 0.0) <= 0.90
        and int(row.get("train_blocks_positive", 0) or 0) >= 4
        and int(row.get("validation_blocks_positive", 0) or 0) >= 4
    )


def _near_final(row: pd.Series) -> bool:
    return (
        float(row.get("train_calmar", 0.0) or 0.0) >= 1.0
        and float(row.get("validation_calmar", 0.0) or 0.0) >= 1.0
        and abs(float(row.get("train_mdd", 0.0) or 0.0)) <= 0.35
        and abs(float(row.get("validation_mdd", 0.0) or 0.0)) <= 0.35
        and int(row.get("walkforward_passes", 0) or 0) >= 3
    )


def _shootout_score(row: pd.Series) -> float:
    return float(
        (100.0 if bool(row.get("quick_pass")) else 0.0)
        + (250.0 if bool(row.get("near_final")) else 0.0)
        + (500.0 if bool(row.get("accepted")) else 0.0)
        + 8.0 * float(row.get("robust_passes", 0) or 0)
        + 4.0 * float(row.get("walkforward_passes", 0) or 0)
        + 2.0 * float(row.get("train_calmar", 0.0) or 0.0)
        + 2.0 * float(row.get("validation_calmar", 0.0) or 0.0)
        - abs(float(row.get("train_mdd", 0.0) or 0.0)) * 4.0
        - abs(float(row.get("validation_mdd", 0.0) or 0.0)) * 4.0
    )


def _method_table(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty:
        return pd.DataFrame()
    rows = []
    for method, frame in leaderboard.groupby("shootout_method"):
        best = frame.sort_values("shootout_score", ascending=False).iloc[0]
        rows.append(
            {
                "method": method,
                "tested": int(len(frame)),
                "quick_pass": int(frame["quick_pass"].sum()),
                "near_final": int(frame["near_final"].sum()),
                "accepted_strict": int(frame["accepted"].sum()) if "accepted" in frame else 0,
                "best_candidate": best.get("candidate_id", ""),
                "best_train_calmar": float(best.get("train_calmar", 0.0) or 0.0),
                "best_validation_calmar": float(best.get("validation_calmar", 0.0) or 0.0),
                "best_robust_passes": int(best.get("robust_passes", 0) or 0),
                "best_walkforward_passes": int(best.get("walkforward_passes", 0) or 0),
                "best_score": float(best.get("shootout_score", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("best_score", ascending=False)


def _json_clean(value):
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
