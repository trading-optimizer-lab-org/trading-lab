from pathlib import Path

import pandas as pd

from trading_lab.results import merge_leaderboards
from trading_lab.survival import (
    SurvivalCriteria,
    build_survival_grid,
    encode_feature_spec,
    evaluate_survival_candidate,
    expand_feature_parameter_space,
    public_feature_columns,
    split_train_validation,
    survival_score,
    validate_spy_only_candidate,
)
from trading_lab.survival import _run_candidate, _signals_for_rule


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


def test_expand_feature_parameter_space_uses_all_public_feature_columns() -> None:
    data = sample_daily_data()
    data["qqq_ret_20"] = 0.01
    data["hyg_ret_20"] = -0.01
    parameter_space = {
        "rule": ["feature_threshold"],
        "feature_name": ["__ALL_PUBLIC_FEATURES__"],
        "feature_threshold": [0.0],
    }

    expanded = expand_feature_parameter_space(parameter_space, data)

    assert expanded["feature_name"] == ["qqq_ret_20", "hyg_ret_20"]
    assert public_feature_columns(data) == ["qqq_ret_20", "hyg_ret_20"]


def test_evaluate_survival_candidate_supports_feature_rules() -> None:
    data = sample_daily_data()
    data["qqq_ret_20"] = data["close"].pct_change(20).fillna(0.0)

    row = evaluate_survival_candidate(
        data,
        {"rule": "feature_threshold", "feature_name": "qqq_ret_20", "feature_threshold": 0.0},
        initial_cash=10_000,
        commission_bps=0,
        slippage_bps=0,
    )

    assert row["feature_count"] == 1
    assert row["locked_opened"] is False


def test_evaluate_survival_candidate_supports_feature_vote_combo() -> None:
    data = sample_daily_data()
    data["lqd_ret_63"] = data["close"].pct_change(63).fillna(0.0)
    data["spy_vs_tnx_ret_63"] = data["close"].pct_change(63).fillna(0.0)
    specs = ";".join(
        [
            encode_feature_spec(name="lqd_ret_63", kind="threshold", value=0.0),
            encode_feature_spec(name="spy_vs_tnx_ret_63", kind="threshold", value=0.0),
        ]
    )

    row = evaluate_survival_candidate(
        data,
        {"rule": "feature_vote", "feature_specs": specs, "min_votes": 2},
        initial_cash=10_000,
        commission_bps=0,
        slippage_bps=0,
    )

    assert row["feature_count"] == 2
    assert row["locked_opened"] is False


def test_evaluate_survival_candidate_supports_feature_score_combo() -> None:
    data = sample_daily_data()
    data["lqd_ret_63"] = data["close"].pct_change(63).fillna(0.0)
    data["spy_vs_tnx_ret_63"] = data["close"].pct_change(63).fillna(0.0)
    specs = ";".join(
        [
            encode_feature_spec(name="lqd_ret_63", kind="zscore", value=0.0, window=60),
            encode_feature_spec(name="spy_vs_tnx_ret_63", kind="zscore", value=0.0, window=60),
        ]
    )

    row = evaluate_survival_candidate(
        data,
        {"rule": "feature_score", "feature_specs": specs, "score_threshold": 0.0},
        initial_cash=10_000,
        commission_bps=0,
        slippage_bps=0,
    )

    assert row["feature_count"] == 2
    assert row["locked_opened"] is False


def test_evaluate_survival_candidate_supports_long_short_feature_vote() -> None:
    data = sample_daily_data()
    data["lqd_ret_63"] = data["close"].pct_change(63).fillna(0.0)
    data["vix_ret_21"] = -data["close"].pct_change(21).fillna(0.0)
    long_specs = encode_feature_spec(name="lqd_ret_63", kind="threshold", value=0.0)
    short_specs = encode_feature_spec(name="vix_ret_21", kind="threshold", value=0.0)

    row = evaluate_survival_candidate(
        data,
        {
            "rule": "feature_vote_position",
            "long_specs": long_specs,
            "short_specs": short_specs,
            "long_min_votes": 1,
            "short_min_votes": 1,
        },
        initial_cash=10_000,
        commission_bps=0,
        slippage_bps=0,
    )

    assert row["feature_count"] == 2
    assert row["locked_opened"] is False


def test_spy_only_always_invested_rule_never_goes_to_cash() -> None:
    data = sample_daily_data()
    data["slow_trend"] = data["close"].pct_change(63).fillna(0.0)
    data["stress"] = -data["close"].pct_change(21).fillna(0.0)
    long_specs = encode_feature_spec(name="slow_trend", kind="threshold", value=0.0)
    short_specs = encode_feature_spec(name="stress", kind="threshold", value=0.0)

    signals = evaluate_survival_candidate(
        data,
        {
            "rule": "spy_long_short_always",
            "long_specs": long_specs,
            "short_specs": short_specs,
            "long_min_votes": 1,
            "short_min_votes": 1,
        },
        initial_cash=10_000,
        commission_bps=0,
        slippage_bps=0,
    )

    assert signals["traded_asset"] == "SPY"
    assert signals["cash_allowed"] is False
    assert signals["always_fully_invested"] is True
    assert signals["feature_count"] == 2
    assert signals["locked_opened"] is False


def test_spy_only_backtest_starts_in_spy_position_not_cash() -> None:
    data = sample_daily_data()
    data["slow_trend"] = data["close"].pct_change(63).fillna(0.0)
    data["stress"] = -data["close"].pct_change(21).fillna(0.0)
    params = {
        "rule": "spy_long_short_always",
        "long_specs": encode_feature_spec(name="slow_trend", kind="threshold", value=0.0),
        "short_specs": encode_feature_spec(name="stress", kind="threshold", value=0.0),
        "long_min_votes": 1,
        "short_min_votes": 1,
        "confirm_days": 3,
        "min_hold_days": 5,
    }

    result = _run_candidate(data, params, 10_000, 0, 0)

    assert not result.trades.empty
    assert result.trades.iloc[0]["entry_time"] == data.index[0]


def test_spy_only_position_memory_avoids_single_day_flip() -> None:
    data = sample_daily_data()
    data["long_feature"] = 1.0
    data["short_feature"] = -1.0
    data.loc[data.index[10], "short_feature"] = 1.0
    params = {
        "rule": "spy_long_short_always",
        "long_specs": encode_feature_spec(name="long_feature", kind="threshold", value=0.0),
        "short_specs": encode_feature_spec(name="short_feature", kind="threshold", value=0.0),
        "long_min_votes": 1,
        "short_min_votes": 1,
        "confirm_days": 3,
        "min_hold_days": 5,
    }

    signal = _signals_for_rule(data, params)

    assert set(signal.unique()) == {1}


def test_spy_only_score_rule_is_always_long_or_short_spy() -> None:
    data = sample_daily_data()
    data["risk_feature"] = data["close"].pct_change(21).fillna(0.0)
    params = {
        "rule": "spy_long_short_score",
        "feature_specs": encode_feature_spec(name="risk_feature", kind="zscore", value=0.0, window=20),
        "score_threshold": 0.25,
        "confirm_days": 2,
        "min_hold_days": 3,
    }

    row = evaluate_survival_candidate(data, params, initial_cash=10_000, commission_bps=0, slippage_bps=0)
    signal = _signals_for_rule(data, params)

    assert row["traded_asset"] == "SPY"
    assert row["cash_allowed"] is False
    assert row["always_fully_invested"] is True
    assert set(signal.unique()) <= {-1, 1}


def test_validate_spy_only_candidate_rejects_cash_and_other_assets() -> None:
    validate_spy_only_candidate({"rule": "spy_long_short_always", "long_specs": "a", "short_specs": "b"})

    try:
        validate_spy_only_candidate({"rule": "portfolio_regime", "safe_asset": "CASH"})
    except ValueError as exc:
        assert "SPY-only" in str(exc)
    else:
        raise AssertionError("portfolio_regime must be invalid for SPY-only objective")


def test_evaluate_survival_candidate_supports_portfolio_regime() -> None:
    data = sample_daily_data()
    data["tlt_close_ratio"] = 1.0 + data["close"].pct_change().fillna(0.0).mul(-0.2).cumsum()
    data["gld_close_ratio"] = 1.0 + data["close"].pct_change().fillna(0.0).mul(0.1).cumsum()
    data["lqd_ret_63"] = data["close"].pct_change(63).fillna(0.0)
    data["vix_ret_21"] = -data["close"].pct_change(21).fillna(0.0)

    row = evaluate_survival_candidate(
        data,
        {
            "rule": "portfolio_regime",
            "risk_on_specs": encode_feature_spec(name="lqd_ret_63", kind="threshold", value=0.0),
            "stress_specs": encode_feature_spec(name="vix_ret_21", kind="threshold", value=0.0),
            "risk_on_min_votes": 1,
            "stress_min_votes": 1,
            "risk_asset": "SPY",
            "safe_asset": "TLT",
            "stress_asset": "GLD",
        },
        initial_cash=10_000,
        commission_bps=0,
        slippage_bps=0,
    )

    assert row["feature_count"] == 2
    assert row["locked_opened"] is False


def test_merge_survival_leaderboard_preserves_best_score(tmp_path: Path) -> None:
    first = tmp_path / "survival_stage_0.csv"
    second = tmp_path / "survival_stage_1.csv"
    pd.DataFrame([{"candidate_id": "a", "survival_score": 1.0}]).to_csv(first, index=False)
    pd.DataFrame([{"candidate_id": "b", "survival_score": 3.0}]).to_csv(second, index=False)

    merged = merge_leaderboards([first, second], score_column="survival_score")

    assert merged.iloc[0]["candidate_id"] == "b"
