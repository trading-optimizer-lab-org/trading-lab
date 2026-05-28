from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import math
import pandas as pd
import pytest

from scripts.merge_weekly_spy_sharpe_10methods_9h import main as merge_main
from scripts.merge_weekly_spy_sharpe_4methods_180 import merge_outputs
from scripts.run_weekly_spy_sharpe_4methods_180_stage import (
    _aurora_candidate_to_row,
    _aurora_price_and_pending_frames,
    _spy_common_daily,
)
from scripts.merge_weekly_spy_sharpe_10methods_9h_state import merge_state_files
from trading_lab.monthly_risk import MonthlyRiskSearchConfig
from trading_lab.weekly_7methods_stateful import _candidate_from_hpo_config, _hpo_configspace, run_weekly_machine_learning_search
from trading_lab.weekly_multi_asset import WeeklyMachineLearningCandidate


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


def test_merge_positive_years_mode_does_not_use_validation_for_verified(tmp_path: Path) -> None:
    stage = tmp_path / "wave_1" / "beam" / "stage_0"
    stage.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "candidate_id": "train_ok_validation_bad",
                "method": "beam",
                "wave": 1,
                "stage": 0,
                "weekly_multi_asset_score": 1_700_000,
                "train_sharpe": 1.7,
                "validation_sharpe": -0.5,
                "train_cagr": 0.08,
                "train_years_positive": 12,
                "validation_years_positive": 2,
                "average_abs_exposure": 0.5,
                "locked_opened": False,
            }
        ]
    ).to_csv(stage / "weekly_positive_leaderboard_stage_beam_1_0.csv", index=False)
    (stage / "job_meta.json").write_text(
        json.dumps({"started_epoch": 1000, "ended_epoch": 1100, "method": "beam", "stage": 0, "wave": 1}),
        encoding="utf-8",
    )

    summary = merge_outputs(
        input_glob=str(tmp_path / "**" / "weekly_positive_leaderboard_stage_*.csv"),
        output_dir=tmp_path / "merged",
        file_prefix="weekly_positive",
        expected_jobs=1,
        expected_methods=["beam"],
        score_mode="train_sharpe_positive_years_report_validation",
    )
    verified = pd.read_csv(tmp_path / "merged" / "weekly_positive_verified.csv")

    assert summary["score_mode"] == "train_sharpe_positive_years_report_validation"
    assert list(verified["candidate_id"]) == ["train_ok_validation_bad"]


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


def test_real_hpo_configspace_handles_single_and_many_asset_universes() -> None:
    pytest.importorskip("ConfigSpace")

    config = MonthlyRiskSearchConfig(stage=0, total_stages=1, max_features=2, seed_pool=8)
    single = _hpo_configspace(["spec_a"], [("SPY",)], config)
    single_candidate = _candidate_from_hpo_config(single.get_default_configuration(), ["spec_a"], [("SPY",)], config)

    assert single_candidate.assets == ("SPY",)
    assert single_candidate.specs == ("spec_a",)

    many_universes = [("SPY",), ("SPY", "QQQ"), ("SPY", "TLT", "GLD")]
    many = _hpo_configspace(["spec_a", "spec_b", "spec_c"], many_universes, config)
    many_candidate = _candidate_from_hpo_config(many.sample_configuration(), ["spec_a", "spec_b", "spec_c"], many_universes, config)

    assert many_candidate.assets
    assert set(many_candidate.assets).issubset({"SPY", "QQQ", "TLT", "GLD"})
    assert many_candidate.specs


def test_aurora_candidate_uses_clear_metric_proxy_instead_of_silent_nan() -> None:
    args = Namespace(wave=1, stage=2, time_budget_minutes=1)
    row = _aurora_candidate_to_row(
        {
            "candidate_id": "ml-1",
            "model": "forest",
            "feature_set": ["ret_lag_1"],
            "train_metrics": {"cagr": 12.0, "mdd": -6.0, "calmar": 2.0},
            "validation_metrics": {"cagr": 8.0, "mdd": -5.0, "calmar": 1.6},
        },
        args,
    )

    assert math.isfinite(row["train_sharpe"])
    assert math.isfinite(row["validation_sharpe"])
    assert row["aurora_metric_source"] == "train:calmar_proxy;validation:calmar_proxy"
    assert row["aurora_comparable_sharpe"] is False
    assert row["verified_sharpe_robust"] is False


def test_spy_common_daily_preserves_multi_asset_ratio_columns() -> None:
    frame = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [100],
            "qqq_close_ratio": [1.2],
        }
    )

    out = _spy_common_daily(frame)

    assert "qqq_close_ratio" in out.columns
    assert "adj_close" in out.columns


def test_aurora_price_and_pending_frames_split_panel_columns() -> None:
    frame = pd.DataFrame(
        {
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "adj_close": [1.5],
            "volume": [100],
            "qqq_close_ratio": [1.2],
            "tlt_ret_21": [0.03],
        }
    )

    price, pending = _aurora_price_and_pending_frames(frame)

    assert list(price.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert set(pending.columns) == {"qqq_close_ratio", "tlt_ret_21"}


def test_weekly_machine_learning_search_checks_deadline_inside_candidate_batch(monkeypatch) -> None:
    weeks = pd.date_range("1994-01-07", "2019-12-27", freq="W-FRI")
    examples = pd.DataFrame(
        {
            "decision_date": weeks,
            "target_week_end": weeks,
            "target_year": weeks.year,
            "spy_return_next_week": 0.01,
            "sp500_down_year": False,
            "asset_spy_return_next_week": 0.01,
            "feature_a": range(len(weeks)),
            "feature_b": range(len(weeks), 0, -1),
        }
    )
    calls = {"count": 0}

    def fake_next_ml_candidates(*args, **kwargs):
        return [
            WeeklyMachineLearningCandidate(features=("feature_a",), assets=("SPY",), model="ridge"),
            WeeklyMachineLearningCandidate(features=("feature_b",), assets=("SPY",), model="ridge"),
        ]

    def fake_evaluate(*args, **kwargs):
        calls["count"] += 1
        row = {
            "candidate_id": f"ml_{calls['count']}",
            "method": "machine_learning",
            "features": "feature_a",
            "weekly_multi_asset_score": float(calls["count"]),
            "train_sharpe": 1.2,
            "validation_sharpe": -1.0,
            "train_cagr": 0.08,
            "train_years_positive": 12,
            "validation_years_positive": 1,
            "average_abs_exposure": 0.5,
            "accepted": True,
            "verified_sharpe_robust": True,
        }
        return row, pd.DataFrame(), pd.DataFrame()

    ticks = iter([0.0, 0.0, 2.0, 2.0])
    monkeypatch.setattr("trading_lab.weekly_7methods_stateful._next_ml_candidates", fake_next_ml_candidates)
    monkeypatch.setattr("trading_lab.weekly_7methods_stateful.evaluate_weekly_machine_learning_candidate", fake_evaluate)
    monkeypatch.setattr("trading_lab.weekly_7methods_stateful.time.monotonic", lambda: next(ticks, 2.0))

    rows, _ = run_weekly_machine_learning_search(
        examples,
        MonthlyRiskSearchConfig(stage=0, total_stages=1, seed_pool=10, top_rows_per_stage=10),
        method="machine_learning",
        wave=1,
        stage=0,
        time_budget_minutes=0.001,
    )

    assert len(rows) == 1
    assert calls["count"] == 1
