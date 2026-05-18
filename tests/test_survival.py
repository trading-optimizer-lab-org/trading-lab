from pathlib import Path

import pandas as pd

from trading_lab.results import merge_leaderboards
from trading_lab.survival import (
    SurvivalCriteria,
    build_survival_grid,
    evaluate_survival_candidate,
    split_train_validation,
    survival_score,
)


def sample_daily_data() -> pd.DataFrame:
    index = pd.date_range("2010-01-01", periods=2800, freq="B", name="timestamp")
    close = [100 + index_ * 0.4 + (index_ % 7) for index_ in range(len(index))]
    return pd.DataFrame(
        {
            "open": close,
            "high": [price + 1 for price in close],
            "low": [price - 1 for price in close],
            "close": close,
            "volume": [1000] * len(close),
        },
        index=index,
    )


def test_split_train_validation_keeps_locked_out() -> None:
    data = pd.DataFrame(index=pd.to_datetime(["2013-10-18", "2013-10-21", "2019-12-31", "2020-01-02"]))

    train, validation = split_train_validation(data)

    assert train.index.max() <= pd.Timestamp("2013-10-18")
    assert validation.index.min() >= pd.Timestamp("2013-10-21")
    assert validation.index.max() <= pd.Timestamp("2019-12-31")


def test_survival_score_penalizes_train_validation_gap() -> None:
    stable = {"train_calmar": 1.5, "validation_calmar": 1.4, "feature_count": 2}
    unstable = {"train_calmar": 7.5, "validation_calmar": 1.4, "feature_count": 2}

    assert survival_score(stable) > survival_score(unstable)


def test_survival_rejects_excessive_train_calmar() -> None:
    criteria = SurvivalCriteria(max_train_calmar=1.0)
    row = {
        "train_calmar": 2.0,
        "validation_calmar": 2.0,
        "train_cagr": 0.2,
        "validation_cagr": 0.2,
        "train_mdd": -0.1,
        "validation_mdd": -0.1,
        "trades_per_year": 20.0,
        "long_fraction": 0.5,
        "validation_negative_years": 0,
        "feature_count": 2,
    }

    assert criteria.rejection_reason(row) == "train_calmar_too_high"


def test_survival_counts_passed_robust_filters() -> None:
    criteria = SurvivalCriteria()
    row = {
        "train_calmar": 1.5,
        "validation_calmar": 1.4,
        "train_cagr": 0.08,
        "validation_cagr": 0.07,
        "train_mdd": -0.2,
        "validation_mdd": -0.2,
        "trades_per_year": 20.0,
        "long_fraction": 0.5,
        "validation_negative_years": 0,
        "feature_count": 2,
    }

    assert criteria.pass_count(row) == len(criteria.checks(row))


def test_build_survival_grid_splits_across_stages() -> None:
    grid = build_survival_grid({"fast_window": [2, 3], "slow_window": [5, 8]}, stage=1, total_stages=2)

    assert grid == [
        {"rule": "ma_crossover", "fast_window": 2, "slow_window": 8},
        {"rule": "ma_crossover", "fast_window": 3, "slow_window": 8},
    ]


def test_build_survival_grid_can_search_multiple_rule_families() -> None:
    grid = build_survival_grid(
        {
            "rule": ["ma_crossover", "rsi_reversion", "mean_reversion", "linear_score"],
            "fast_window": [2],
            "slow_window": [5],
            "rsi_window": [2],
            "rsi_buy": [30],
            "rsi_sell": [60],
            "reversion_window": [10],
            "entry_zscore": [1.0],
            "exit_zscore": [0.0],
            "fast_return_window": [5],
            "slow_return_window": [60],
            "risk_window": [20],
            "score_threshold": [0.0],
        },
        stage=0,
        total_stages=1,
    )

    assert {candidate["rule"] for candidate in grid} == {
        "ma_crossover",
        "rsi_reversion",
        "mean_reversion",
        "linear_score",
    }


def test_evaluate_survival_candidate_reports_locked_closed() -> None:
    row = evaluate_survival_candidate(
        sample_daily_data(),
        {"rule": "ma_crossover", "fast_window": 2, "slow_window": 5},
        initial_cash=10_000,
        commission_bps=0,
        slippage_bps=0,
    )

    assert row["locked_opened"] is False
    assert "train_calmar" in row
    assert "validation_calmar" in row
    assert "robust_passes" in row


def test_evaluate_survival_candidate_supports_non_ma_rules() -> None:
    row = evaluate_survival_candidate(
        sample_daily_data(),
        {"rule": "rsi_reversion", "rsi_window": 2, "rsi_buy": 30, "rsi_sell": 60},
        initial_cash=10_000,
        commission_bps=0,
        slippage_bps=0,
    )

    assert row["candidate_id"].startswith("rsi_buy_30")
    assert row["locked_opened"] is False


def test_evaluate_survival_candidate_supports_linear_score_rule() -> None:
    row = evaluate_survival_candidate(
        sample_daily_data(),
        {
            "rule": "linear_score",
            "fast_return_window": 5,
            "slow_return_window": 60,
            "risk_window": 20,
            "score_threshold": 0.0,
        },
        initial_cash=10_000,
        commission_bps=0,
        slippage_bps=0,
    )

    assert row["feature_count"] == 4
    assert row["locked_opened"] is False


def test_merge_survival_leaderboard_preserves_best_score(tmp_path: Path) -> None:
    first = tmp_path / "survival_stage_0.csv"
    second = tmp_path / "survival_stage_1.csv"
    pd.DataFrame([{"candidate_id": "a", "survival_score": 1.0}]).to_csv(first, index=False)
    pd.DataFrame([{"candidate_id": "b", "survival_score": 3.0}]).to_csv(second, index=False)

    merged = merge_leaderboards([first, second], score_column="survival_score")

    assert merged.iloc[0]["candidate_id"] == "b"
