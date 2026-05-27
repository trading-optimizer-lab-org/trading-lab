from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.merge_weekly_spy_sharpe_10methods_9h import main as merge_main
from scripts.merge_weekly_spy_sharpe_10methods_9h_state import merge_state_files


def test_merge_10methods_outputs_partial_summary_and_sharpe_files(tmp_path: Path, monkeypatch) -> None:
    stage_a = tmp_path / "wave_1" / "beam" / "stage_0"
    stage_b = tmp_path / "wave_1" / "github_ml" / "stage_0"
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
                "locked_opened": False,
                "elapsed_seconds": 85.0,
            }
        ]
    ).to_csv(stage_a / "weekly_spy_sharpe_10methods_9h_leaderboard_stage_beam_1_0.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "github_relaxed",
                "method": "github_ml",
                "wave": 1,
                "stage": 0,
                "weekly_multi_asset_score": 250_000,
                "train_sharpe": 0.5,
                "validation_sharpe": 0.4,
                "locked_opened": False,
                "elapsed_seconds": 85.0,
            }
        ]
    ).to_csv(stage_b / "weekly_spy_sharpe_10methods_9h_leaderboard_stage_github_ml_1_0.csv", index=False)
    for folder, method in ((stage_a, "beam"), (stage_b, "github_ml")):
        (folder / "job_meta.json").write_text(
            json.dumps({"started_epoch": 1000, "ended_epoch": 1300, "method": method, "stage": 0, "wave": 1}),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "sys.argv",
        [
            "merge",
            "--input-glob",
            str(tmp_path / "**" / "weekly_spy_sharpe_10methods_9h_leaderboard_stage_*.csv"),
            "--output-dir",
            str(tmp_path / "merged"),
            "--config",
            "configs/weekly_spy_sharpe_10methods_9h_waves.yaml",
            "--file-prefix",
            "weekly_spy_sharpe_10methods_9h",
            "--expected-jobs",
            "900",
        ],
    )
    assert merge_main() == 0

    summary = json.loads((tmp_path / "merged" / "weekly_spy_sharpe_10methods_9h_summary.json").read_text(encoding="utf-8"))
    verified = pd.read_csv(tmp_path / "merged" / "weekly_spy_sharpe_10methods_9h_verified.csv")
    relaxed = pd.read_csv(tmp_path / "merged" / "weekly_spy_sharpe_10methods_9h_relaxed_valid.csv")

    assert summary["artifact"] == "weekly-spy-sharpe-10methods-9h-waves-leaderboard"
    assert summary["artifacts_downloaded"] == 2
    assert summary["expected_artifacts"] == 900
    assert summary["partial"] is True
    assert list(verified["candidate_id"]) == ["beam_ok"]
    assert set(relaxed["candidate_id"]) == {"beam_ok", "github_relaxed"}


def test_merge_10methods_state_keeps_methods_even_when_partial(tmp_path: Path) -> None:
    state_path = tmp_path / "state" / "weekly_spy_sharpe_10methods_9h_state_beam_1_0.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "method": "beam",
                "candidates": [
                    {"candidate_id": "a", "train_score": 1.0},
                    {"candidate_id": "a", "train_score": 2.0},
                ],
                "locked_opened": False,
                "validation_role": "report_only",
            }
        ),
        encoding="utf-8",
    )

    summary = merge_state_files(
        [str(state_path)],
        tmp_path / "merged_state",
        methods=["beam", "genetic"],
        state_top=500,
        expected_files_per_method=18,
        allow_missing_files_per_method=18,
        file_prefix="weekly_spy_sharpe_10methods_9h",
    )

    assert summary["state_files_by_method"] == {"beam": 1, "genetic": 0}
    beam_state = json.loads((tmp_path / "merged_state" / "state" / "beam.json").read_text(encoding="utf-8"))
    genetic_state = json.loads((tmp_path / "merged_state" / "state" / "genetic.json").read_text(encoding="utf-8"))
    assert beam_state["candidate_count"] == 1
    assert beam_state["candidates"][0]["train_score"] == 2.0
    assert genetic_state["candidate_count"] == 0
