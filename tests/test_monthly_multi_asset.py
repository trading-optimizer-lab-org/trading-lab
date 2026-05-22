from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_lab.monthly_multi_asset import (
    MonthlyMultiAssetCandidate,
    MonthlyRiskSearchConfig,
    available_tradable_assets,
    build_monthly_multi_asset_examples,
    evaluate_monthly_multi_asset_candidate,
    merge_monthly_multi_asset_leaderboards,
    run_monthly_multi_asset_search,
    write_monthly_multi_asset_outputs,
)


def _multi_asset_daily_sample() -> pd.DataFrame:
    dates = pd.bdate_range("1993-01-29", "2021-03-31")
    index = range(len(dates))
    close = [100.0 + value * 0.05 + (value % 17) * 0.12 for value in index]
    qqq_ratio = [1.0 + value * 0.00008 for value in index]
    tlt_ratio = [1.0 + ((value % 250) - 125) * 0.0005 for value in index]
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close,
            "high": [value * 1.01 for value in close],
            "low": [value * 0.99 for value in close],
            "close": close,
            "volume": [1000] * len(dates),
            "qqq_close_ratio": qqq_ratio,
            "tlt_close_ratio": tlt_ratio,
        }
    )
    return data.set_index("timestamp")


def test_multi_asset_lists_tradeable_assets_with_coverage() -> None:
    assets = available_tradable_assets(_multi_asset_daily_sample())

    names = {row["asset"] for row in assets}
    assert {"SPY", "QQQ", "TLT"}.issubset(names)


def test_multi_asset_examples_include_rotatable_asset_returns() -> None:
    examples = build_monthly_multi_asset_examples(_multi_asset_daily_sample(), start_year=1994, end_year=1994)

    assert "asset_spy_return_next_month" in examples
    assert "asset_qqq_return_next_month" in examples
    assert "asset_qqq_mom_6m" in examples


def test_multi_asset_candidate_rotates_assets_and_never_opens_locked() -> None:
    examples = build_monthly_multi_asset_examples(_multi_asset_daily_sample(), start_year=2006, end_year=2021)
    candidate = MonthlyMultiAssetCandidate(
        ("month_number|threshold|0|1|1",),
        ("SPY", "QQQ", "TLT"),
        selector="momentum_3m",
        intercept=0.5,
        scale=0.25,
    )

    row, positions, year_by_year = evaluate_monthly_multi_asset_candidate(examples, candidate)

    assert row["locked_opened"] is False
    assert positions["period"].isin(["locked"]).sum() == 0
    assert year_by_year["period"].isin(["locked"]).sum() == 0
    assert positions["traded_asset"].isin(["SPY", "QQQ", "TLT"]).all()
    assert row["min_exposure"] >= -1.0
    assert row["max_exposure"] <= 1.0


def test_multi_asset_search_and_outputs(tmp_path: Path) -> None:
    examples = build_monthly_multi_asset_examples(_multi_asset_daily_sample(), start_year=1994, end_year=2012)
    rows = run_monthly_multi_asset_search(
        examples,
        MonthlyRiskSearchConfig(stage=0, total_stages=2, seed_pool=10, beam_width=3, generations=1),
    )

    write_monthly_multi_asset_outputs(rows, examples, tmp_path, stage=0)

    assert rows
    assert "assets" in rows[0]
    assert "asset_selector" in rows[0]
    assert (tmp_path / "monthly_multi_asset_leaderboard.csv").exists()
    assert (tmp_path / "monthly_multi_asset_monthly_positions.csv").exists()
    assert (tmp_path / "monthly_multi_asset_year_by_year.csv").exists()


def test_multi_asset_merge_counts_verified_train_validation_only(tmp_path: Path) -> None:
    stage = tmp_path / "stage.csv"
    pd.DataFrame(
        [
            {
                "candidate_id": "ok",
                "monthly_multi_asset_score": 10.0,
                "accepted": True,
                "train_years_ge_10pct": 14,
                "train_years_total": 14,
                "validation_years_ge_10pct": 12,
                "validation_years_total": 12,
                "locked_opened": False,
            },
            {
                "candidate_id": "fails_validation",
                "monthly_multi_asset_score": 9.0,
                "accepted": True,
                "train_years_ge_10pct": 14,
                "train_years_total": 14,
                "validation_years_ge_10pct": 11,
                "validation_years_total": 12,
                "locked_opened": False,
            },
        ]
    ).to_csv(stage, index=False)

    summary = merge_monthly_multi_asset_leaderboards([stage], tmp_path / "merged")

    verified = pd.read_csv(tmp_path / "merged" / "monthly_multi_asset_train_validation_10pct.csv")
    assert summary["unique_verified_train_validation_10pct"] == 1
    assert verified["candidate_id"].tolist() == ["ok"]
