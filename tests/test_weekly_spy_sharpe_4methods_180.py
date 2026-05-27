from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.merge_weekly_spy_sharpe_4methods_180 import merge_outputs


def test_merge_outputs_uses_exact_sharpe_rules_and_parallelism(tmp_path: Path) -> None:
    stage_a = tmp_path / "beam" / "stage_0"
    stage_b = tmp_path / "genetic" / "stage_0"
    stage_a.mkdir(parents=True)
    stage_b.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "candidate_id": "beam_ok",
                "method": "beam",
                "wave": 1,
                "stage": 0,
                "weekly_multi_asset_score": 1_500_000,
                "train_sharpe": 1.5,
                "validation_sharpe": 1.3,
                "train_cagr": 0.02,
                "validation_cagr": 0.01,
                "train_mdd": -0.10,
                "validation_mdd": -0.12,
                "locked_opened": False,
                "elapsed_seconds": 10.0,
            }
        ]
    ).to_csv(stage_a / "weekly_spy_sharpe_4methods_180_leaderboard_stage_0.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "genetic_relaxed",
                "method": "genetic",
                "wave": 1,
                "stage": 0,
                "weekly_multi_asset_score": 500_000,
                "train_sharpe": 0.5,
                "validation_sharpe": 0.4,
                "train_cagr": 0.20,
                "validation_cagr": 0.15,
                "train_mdd": -0.10,
                "validation_mdd": -0.12,
                "locked_opened": False,
                "elapsed_seconds": 20.0,
            }
        ]
    ).to_csv(stage_b / "weekly_spy_sharpe_4methods_180_leaderboard_stage_0.csv", index=False)
    for folder in (stage_a, stage_b):
        (folder / "job_meta.json").write_text(
            json.dumps(
                {
                    "started_epoch": 1000,
                    "ended_epoch": 1600,
                    "method": folder.parent.name,
                    "stage": 0,
                }
            ),
            encoding="utf-8",
        )

    summary = merge_outputs(
        input_glob=str(tmp_path / "**" / "weekly_spy_sharpe_4methods_180_leaderboard_stage_*.csv"),
        output_dir=tmp_path / "merged",
        file_prefix="weekly_spy_sharpe_4methods_180",
        expected_jobs=180,
        expected_methods=("beam", "genetic", "aurora_ml", "github_ml"),
    )

    verified = pd.read_csv(tmp_path / "merged" / "weekly_spy_sharpe_4methods_180_verified.csv")
    relaxed = pd.read_csv(tmp_path / "merged" / "weekly_spy_sharpe_4methods_180_relaxed_valid.csv")
    parallelism = json.loads((tmp_path / "merged" / "weekly_spy_sharpe_4methods_180_parallelism.json").read_text(encoding="utf-8"))

    assert summary["verified_sharpe_robust"] == 1
    assert summary["relaxed_valid"] == 2
    assert list(verified["candidate_id"]) == ["beam_ok"]
    assert set(relaxed["candidate_id"]) == {"beam_ok", "genetic_relaxed"}
    assert parallelism["max_parallel_observed"] == 2
    assert parallelism["parallelism_valid"] is False
