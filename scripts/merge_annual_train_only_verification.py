from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge annual train-only stages and verify validation separately.")
    parser.add_argument("--input-glob", default="outputs/annual_sp500_train_only_verify/stages/*.csv")
    parser.add_argument("--output", default="outputs/annual_sp500_train_only_verify/leaderboard.csv")
    parser.add_argument("--verified-output", default="outputs/annual_sp500_train_only_verify/verified_train_validation_100.csv")
    parser.add_argument("--summary", default="outputs/annual_sp500_train_only_verify/summary.json")
    args = parser.parse_args()

    leaderboard = _load_leaderboard(args.input_glob)
    if not leaderboard.empty:
        if "annual_score" not in leaderboard:
            leaderboard["annual_score"] = -1_000_000.0
        leaderboard = leaderboard.sort_values("annual_score", ascending=False)

    verified = _verified_train_validation_100(leaderboard)
    output = Path(args.output)
    verified_output = Path(args.verified_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(output, index=False)
    verified.to_csv(verified_output, index=False)

    summary = summarize_train_only_verification(leaderboard, verified)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def summarize_train_only_verification(leaderboard: pd.DataFrame, verified: pd.DataFrame) -> dict[str, object]:
    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    best_verified = verified.iloc[0].to_dict() if not verified.empty else {}
    return {
        "rows": int(len(leaderboard)),
        "candidates_evaluated": _count_candidates_evaluated(leaderboard),
        "accepted_train_perfect": int(leaderboard["accepted"].fillna(False).astype(bool).sum()) if "accepted" in leaderboard else 0,
        "verified_train_validation_100": int(len(verified)),
        "unique_verified_train_validation_100": int(verified["candidate_id"].nunique()) if "candidate_id" in verified else 0,
        "stage_failures": int(leaderboard["stage_failed"].fillna(False).astype(bool).sum()) if "stage_failed" in leaderboard else 0,
        "score_mode": "train_only_100",
        "validation_role": "report_only",
        "best": _json_clean(best),
        "best_verified": _json_clean(best_verified),
        "locked_opened": False,
    }


def _load_leaderboard(input_glob: str) -> pd.DataFrame:
    paths = sorted(Path(path) for path in glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"no stage files matched {input_glob}")
    frames = [pd.read_csv(path) for path in paths if path.stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _verified_train_validation_100(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty:
        return leaderboard.copy()
    required = {
        "accepted",
        "train_accuracy",
        "validation_accuracy",
        "train_negative_hits",
        "train_negative_total",
        "validation_negative_hits",
        "validation_negative_total",
    }
    missing = required.difference(leaderboard.columns)
    if missing:
        raise ValueError(f"leaderboard missing required verification columns: {sorted(missing)}")
    verified = leaderboard[
        leaderboard["accepted"].fillna(False).astype(bool)
        & (pd.to_numeric(leaderboard["train_accuracy"], errors="coerce") >= 1.0)
        & (pd.to_numeric(leaderboard["validation_accuracy"], errors="coerce") >= 1.0)
        & (
            pd.to_numeric(leaderboard["train_negative_hits"], errors="coerce")
            >= pd.to_numeric(leaderboard["train_negative_total"], errors="coerce")
        )
        & (
            pd.to_numeric(leaderboard["validation_negative_hits"], errors="coerce")
            >= pd.to_numeric(leaderboard["validation_negative_total"], errors="coerce")
        )
    ].copy()
    if "candidate_id" in verified:
        verified = verified.drop_duplicates("candidate_id")
    return verified.sort_values("annual_score", ascending=False) if "annual_score" in verified else verified


def _count_candidates_evaluated(leaderboard: pd.DataFrame) -> int:
    if {"round_name", "stage", "stage_candidates_evaluated"}.issubset(leaderboard.columns):
        return int(
            leaderboard.groupby(["round_name", "stage"])["stage_candidates_evaluated"].max().fillna(0).sum()
        )
    if {"stage", "stage_candidates_evaluated"}.issubset(leaderboard.columns):
        return int(leaderboard.groupby("stage")["stage_candidates_evaluated"].max().fillna(0).sum())
    return int(len(leaderboard))


def _json_clean(value):
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
