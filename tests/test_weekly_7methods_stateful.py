from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from trading_lab.weekly_7methods_stateful import (
    FAIR_5H_WEEKLY_METHODS,
    STATEFUL_WEEKLY_METHODS,
    _engine_method,
    _strict_verified,
    merge_state_files,
)
from trading_lab.weekly_multi_asset import _weekly_calmar_score
from scripts.merge_weekly_7methods_overnight import _partial_summary


def _verified_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": "candidate_a",
        "method": "beam",
        "accepted": True,
        "validation_years_positive": 12,
        "validation_years_total": 12,
        "validation_down_years_ge_5pct": 2,
        "validation_down_years_total": 2,
        "train_years_total": 14,
        "train_down_years_total": 3,
        "locked_opened": False,
        "weekly_multi_asset_score": 100.0,
    }
    row.update(overrides)
    return row


def test_strict_verified_requires_exact_train_validation_counts() -> None:
    rows = pd.DataFrame(
        [
            _verified_row(candidate_id="ok"),
            _verified_row(candidate_id="bad_train_years", train_years_total=13),
            _verified_row(candidate_id="bad_validation_years", validation_years_total=11),
            _verified_row(candidate_id="locked", locked_opened=True),
        ]
    )

    verified = _strict_verified(rows)

    assert verified["candidate_id"].tolist() == ["ok"]


def test_strict_verified_accepts_calmar_similarity_mode() -> None:
    rows = pd.DataFrame(
        [
            _verified_row(
                candidate_id="ok_calmar",
                score_mode="train_calmar_max_validation_80pct_report",
                accepted=True,
                train_calmar=2.0,
                validation_calmar=1.7,
                train_calmar_gt_1=True,
                validation_calmar_gt_1=True,
                validation_calmar_ratio_to_train=0.85,
                validation_calmar_ge_80pct_train=True,
                verified_calmar_similarity=True,
                verified_train_validation_5pct=False,
            ),
            _verified_row(
                candidate_id="bad_ratio",
                score_mode="train_calmar_max_validation_80pct_report",
                accepted=True,
                train_calmar=2.0,
                validation_calmar=1.5,
                train_calmar_gt_1=True,
                validation_calmar_gt_1=True,
                validation_calmar_ratio_to_train=0.75,
                validation_calmar_ge_80pct_train=False,
                verified_calmar_similarity=False,
                verified_train_validation_5pct=False,
            ),
        ]
    )

    verified = _strict_verified(rows)

    assert verified["candidate_id"].tolist() == ["ok_calmar"]


def test_weekly_calmar_score_prioritizes_train_calmar_before_tiebreakers() -> None:
    low_calmar_high_cagr = {
        "train_calmar": 1.5,
        "train_cagr": 0.40,
        "train_mdd": -0.10,
        "train_min_year_return": 0.10,
        "feature_count": 3,
    }
    high_calmar_low_cagr = {
        "train_calmar": 2.0,
        "train_cagr": 0.05,
        "train_mdd": -0.25,
        "train_min_year_return": 0.02,
        "feature_count": 8,
    }

    assert _weekly_calmar_score(high_calmar_low_cagr) > _weekly_calmar_score(low_calmar_high_cagr)


def test_5h_public_methods_map_to_existing_engines() -> None:
    assert len(FAIR_5H_WEEKLY_METHODS) == 7
    assert _engine_method("sobol_random_asha_real") == "real_hpo"
    assert _engine_method("optuna_tpe_hyperband") == "real_hpo"
    assert _engine_method("dehb_real") == "real_hpo"
    assert _engine_method("bohb_real") == "real_hpo"
    assert _engine_method("smac_mf_real") == "real_hpo"
    assert _engine_method("beam") == "beam"
    assert _engine_method("genetic") == "genetic"
    for method in FAIR_5H_WEEKLY_METHODS[:5]:
        assert "lite" not in _engine_method(method)


def test_overnight_partial_summary_counts_missing_wave_stage_artifacts(tmp_path: Path) -> None:
    paths = []
    for name in (
        "weekly_7methods_overnight_leaderboard_stage_beam_1_0.csv",
        "weekly_7methods_overnight_leaderboard_stage_beam_1_1.csv",
        "weekly_7methods_overnight_leaderboard_stage_genetic_2_0.csv",
    ):
        path = tmp_path / name
        path.write_text("candidate_id,method\nx,beam\n", encoding="utf-8")
        paths.append(str(path))

    summary = _partial_summary(paths, methods=["beam", "genetic"], expected_waves=2, expected_stages=2)

    assert summary["artifacts_downloaded"] == 3
    assert summary["expected_artifacts"] == 8
    assert summary["waves_found"] == [1, 2]
    assert summary["stage_files_by_method"] == {"beam": 2, "genetic": 1}
    assert summary["unique_wave_stage_by_method"] == {"beam": 2, "genetic": 1}


def test_state_merge_keeps_train_only_state_and_counts_methods(tmp_path: Path) -> None:
    paths = []
    for method in STATEFUL_WEEKLY_METHODS:
        path = tmp_path / f"{method}.json"
        path.write_text(
            json.dumps(
                {
                    "method": method,
                    "candidates": [
                        {
                            "candidate_id": f"{method}_a",
                            "specs": ["spy_mom_26w_gt_0"],
                            "assets": ["SPY"],
                            "selector": "momentum_26w",
                            "train_score": 10.0,
                            "validation_min_year_return": 0.99,
                        }
                    ],
                    "validation_role": "report_only",
                    "locked_opened": False,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    summary = merge_state_files(paths, tmp_path / "merged", expected_files_per_method=1)

    assert summary["locked_opened"] is False
    assert summary["validation_role"] == "report_only"
    assert summary["state_files_by_method"] == {method: 1 for method in STATEFUL_WEEKLY_METHODS}
    for method in STATEFUL_WEEKLY_METHODS:
        state = json.loads((tmp_path / "merged" / "state" / f"{method}.json").read_text(encoding="utf-8"))
        assert "validation_min_year_return" not in state["candidates"][0]


def test_state_merge_fails_when_method_state_is_missing(tmp_path: Path) -> None:
    path = tmp_path / "beam.json"
    path.write_text(json.dumps({"method": "beam", "candidates": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing weekly 7-method state files"):
        merge_state_files([path], tmp_path / "merged", expected_files_per_method=1)


def test_state_merge_allows_one_missing_runner_per_method(tmp_path: Path) -> None:
    paths = []
    for method in STATEFUL_WEEKLY_METHODS:
        files = 2 if method != "smac_mf_lite" else 1
        for index in range(files):
            path = tmp_path / f"{method}_{index}.json"
            path.write_text(json.dumps({"method": method, "candidates": []}), encoding="utf-8")
            paths.append(path)

    summary = merge_state_files(
        paths,
        tmp_path / "merged",
        expected_files_per_method=2,
        allow_missing_files_per_method=1,
    )

    assert summary["state_files_by_method"]["smac_mf_lite"] == 1
    assert summary["missing_state_files_by_method"] == {"smac_mf_lite": 1}


def test_state_merge_still_fails_when_missing_more_than_allowed(tmp_path: Path) -> None:
    paths = []
    for method in STATEFUL_WEEKLY_METHODS:
        if method == "smac_mf_lite":
            continue
        for index in range(2):
            path = tmp_path / f"{method}_{index}.json"
            path.write_text(json.dumps({"method": method, "candidates": []}), encoding="utf-8")
            paths.append(path)

    with pytest.raises(ValueError, match="smac_mf_lite=0/2"):
        merge_state_files(
            paths,
            tmp_path / "merged",
            expected_files_per_method=2,
            allow_missing_files_per_method=1,
        )
