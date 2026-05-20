from __future__ import annotations

import pandas as pd

from trading_lab.annual_prediction import (
    AnnualBeamConfig,
    AnnualCandidate,
    audit_annual_feature_coverage,
    build_annual_examples,
    load_annual_feature_manifest,
    evaluate_annual_candidate,
    run_annual_beam_search,
)


def _daily_sample() -> pd.DataFrame:
    dates = pd.bdate_range("1979-12-20", "2020-01-10")
    index = range(len(dates))
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [100 + value for value in index],
            "high": [101 + value for value in index],
            "low": [99 + value for value in index],
            "close": [100 + value for value in index],
            "volume": [1000] * len(dates),
            "credit_spread": [2.0 + (value % 7) * 0.1 for value in index],
            "yield_curve": [1.0 - (value % 5) * 0.05 for value in index],
            "cape": [15.0 + (value % 11) * 0.2 for value in index],
            "earnings_yield": [0.06 + (value % 13) * 0.001 for value in index],
            "dividend_yield": [0.02 + (value % 17) * 0.0005 for value in index],
        }
    )
    return data.set_index("timestamp")


def test_build_annual_examples_uses_previous_year_close_only() -> None:
    data = _daily_sample()

    examples = build_annual_examples(data, start_year=1981, end_year=1983)

    assert examples["target_year"].tolist() == [1981, 1982, 1983]
    assert examples["decision_date"].dt.year.tolist() == [1980, 1981, 1982]
    assert "spy_return_next_year" in examples.columns
    assert examples["target_positive"].dtype == bool


def test_build_annual_examples_adds_political_and_valuation_features() -> None:
    examples = build_annual_examples(_daily_sample(), start_year=1981, end_year=1984)

    expected = {
        "presidential_cycle_year",
        "is_election_year",
        "is_post_election_year",
        "president_party",
        "house_control_party",
        "senate_control_party",
        "split_congress",
        "unified_government",
        "cape",
        "earnings_yield",
        "dividend_yield",
    }
    assert expected.issubset(examples.columns)
    assert examples.loc[examples["target_year"] == 1984, "is_election_year"].iloc[0] == 1
    assert examples.loc[examples["target_year"] == 1981, "is_post_election_year"].iloc[0] == 1
    assert examples["cape"].notna().all()


def test_annual_feature_manifest_contains_requested_144_features() -> None:
    manifest = load_annual_feature_manifest()

    assert len(manifest) == 144
    assert manifest["feature"].is_unique
    assert "sp500_return_21d" in set(manifest["feature"])
    assert "consecutive_positive_years" in set(manifest["feature"])


def test_build_annual_examples_exposes_requested_manifest_columns() -> None:
    examples = build_annual_examples(_daily_sample(), start_year=1981, end_year=1984)
    manifest = load_annual_feature_manifest()

    missing = sorted(set(manifest["feature"]) - set(examples.columns))

    assert missing == []


def test_audit_annual_feature_coverage_marks_usable_and_unusable_features() -> None:
    examples = build_annual_examples(_daily_sample(), start_year=1981, end_year=2008)

    audit = audit_annual_feature_coverage(examples)

    assert len(audit) == 144
    assert {"feature", "quality", "usable_in_beam", "first_usable_year"}.issubset(audit.columns)
    assert audit.loc[audit["feature"] == "sp500_return_21d", "usable_in_beam"].iloc[0] in {True, False}
    assert audit.loc[audit["feature"] == "pe_forward", "quality"].iloc[0] == "sin_datos_publicos_fiables"


def test_annual_candidate_evaluates_train_and_validation_without_locked() -> None:
    examples = build_annual_examples(_daily_sample(), start_year=1981, end_year=2020)
    candidate = AnnualCandidate(
        specs=("spy_return_12m|threshold|0|1", "credit_spread_z_3y|threshold|0|-1"),
        min_votes=1,
    )

    row = evaluate_annual_candidate(examples, candidate)

    assert row["locked_opened"] is False
    assert row["train_total"] > 0
    assert row["validation_total"] > 0
    assert row["locked_hits"] == 0
    assert row["locked_total"] == 0
    assert "candidate_id" in row


def test_annual_beam_search_returns_ranked_candidates() -> None:
    examples = build_annual_examples(_daily_sample(), start_year=1981, end_year=2020)
    config = AnnualBeamConfig(
        stage=0,
        total_stages=2,
        seed_pool=12,
        beam_width=4,
        generations=2,
        mutations_per_parent=3,
    )

    rows = run_annual_beam_search(examples, config)

    assert rows
    assert rows[0]["annual_score"] >= rows[-1]["annual_score"]
    assert all(row["locked_opened"] is False for row in rows)
    assert all(int(row["feature_count"]) <= 4 for row in rows)
