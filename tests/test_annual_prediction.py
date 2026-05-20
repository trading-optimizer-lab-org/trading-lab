from __future__ import annotations

import pandas as pd

from scripts.merge_annual_beam_leaderboards import summarize_annual_leaderboard
from trading_lab.annual_prediction import (
    AnnualBeamConfig,
    AnnualCandidate,
    audit_annual_feature_coverage,
    build_annual_examples,
    load_annual_feature_manifest,
    evaluate_annual_candidate,
    run_annual_beam_search,
    score_annual_candidate,
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
    expected_1981 = (
        data.loc[data.index.year == 1981, "close"].iloc[-1]
        / data.loc[data.index.year == 1980, "close"].iloc[-1]
        - 1.0
    )
    assert examples.loc[examples["target_year"] == 1981, "spy_return_next_year"].iloc[0] == expected_1981


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


def test_build_annual_examples_adds_clean_annual_derivatives() -> None:
    examples = build_annual_examples(_daily_sample(), start_year=1981, end_year=1990)

    expected = {
        "cape_annual_change_1y",
        "cape_annual_change_3y",
        "cape_annual_percentile",
        "earnings_yield_annual_change_1y",
    }

    assert expected.issubset(examples.columns)
    assert examples["cape_annual_change_1y"].notna().sum() > 0


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


def test_annual_candidate_supports_range_rules() -> None:
    examples = pd.DataFrame(
        {
            "target_year": [2001, 2002, 2003],
            "target_positive": [False, True, False],
            "spy_return_next_year": [-0.1, 0.2, -0.2],
            "x": [0.1, 0.5, 0.9],
        }
    )
    candidate = AnnualCandidate(("x|range|0.2|0.8|1",), min_votes=1)

    row = evaluate_annual_candidate(examples, candidate, score_mode="train_only_100")

    assert row["train_hits"] == 3


def test_annual_candidate_supports_weighted_or_rules() -> None:
    examples = pd.DataFrame(
        {
            "target_year": [2001, 2002, 2003, 2004],
            "target_positive": [True, False, True, False],
            "spy_return_next_year": [0.1, -0.2, 0.05, -0.1],
            "a": [1.0, 1.0, 0.0, 0.0],
            "b": [1.0, 0.0, 0.0, 0.0],
            "c": [0.0, 0.0, 1.0, 0.0],
        }
    )
    candidate = AnnualCandidate(
        (
            "a|threshold|0.5|1|1",
            "b|threshold|0.5|1|1",
            "c|threshold|0.5|1|2",
        ),
        min_votes=2,
    )

    row = evaluate_annual_candidate(examples, candidate, score_mode="train_only_100")

    assert row["train_hits"] == 4


def test_annual_beam_catalog_includes_range_and_weighted_rules() -> None:
    examples = build_annual_examples(_daily_sample(), start_year=1981, end_year=2020)
    config = AnnualBeamConfig(
        stage=0,
        total_stages=2,
        seed_pool=40,
        beam_width=8,
        generations=2,
        mutations_per_parent=4,
        max_features=6,
    )

    rows = run_annual_beam_search(examples, config)
    specs = ";".join(str(row["specs"]) for row in rows)

    assert "|range|" in specs
    assert any("|2" in part or "|3" in part for part in specs.split(";"))


def test_train_only_100_score_ignores_validation_metrics() -> None:
    base = {
        "feature_count": 2,
        "train_hits": 24,
        "train_total": 25,
        "train_accuracy": 0.96,
        "train_negative_hits": 5,
        "train_negative_total": 6,
        "train_return_mae": 0.10,
    }
    strong_validation = {
        **base,
        "validation_accuracy": 1.0,
        "validation_negative_hits": 3,
        "validation_return_mae": 0.01,
        "always_positive_validation_accuracy": 0.72,
    }
    weak_validation = {
        **base,
        "validation_accuracy": 0.0,
        "validation_negative_hits": 0,
        "validation_return_mae": 0.99,
        "always_positive_validation_accuracy": 0.72,
    }

    assert score_annual_candidate(strong_validation, score_mode="train_only_100") == score_annual_candidate(
        weak_validation,
        score_mode="train_only_100",
    )


def test_train_only_100_acceptance_requires_perfect_train_only() -> None:
    validation_perfect_train_miss = {
        "feature_count": 2,
        "train_hits": 24,
        "train_total": 25,
        "train_accuracy": 0.96,
        "train_negative_hits": 6,
        "train_negative_total": 6,
        "validation_total": 11,
        "validation_accuracy": 1.0,
    }
    train_perfect_validation_bad = {
        "feature_count": 2,
        "train_hits": 25,
        "train_total": 25,
        "train_accuracy": 1.0,
        "train_negative_hits": 6,
        "train_negative_total": 6,
        "validation_total": 11,
        "validation_accuracy": 0.0,
    }

    assert (
        score_annual_candidate(validation_perfect_train_miss, score_mode="train_only_100", field="rejection_reason")
        == "train_not_perfect"
    )
    assert (
        score_annual_candidate(train_perfect_validation_bad, score_mode="train_only_100", field="rejection_reason")
        == ""
    )


def test_train_validation_100_score_ignores_validation_quality_but_acceptance_requires_it() -> None:
    perfect_train_good_validation = {
        "feature_count": 2,
        "train_hits": 25,
        "train_total": 25,
        "train_accuracy": 1.0,
        "train_negative_hits": 6,
        "train_negative_total": 6,
        "train_return_mae": 0.10,
        "validation_hits": 11,
        "validation_total": 11,
        "validation_accuracy": 1.0,
        "validation_negative_hits": 3,
        "validation_negative_total": 3,
    }
    perfect_train_bad_validation = {
        **perfect_train_good_validation,
        "validation_hits": 9,
        "validation_accuracy": 9 / 11,
        "validation_negative_hits": 1,
    }

    assert score_annual_candidate(
        perfect_train_good_validation,
        score_mode="train_validation_100",
    ) == score_annual_candidate(
        perfect_train_bad_validation,
        score_mode="train_validation_100",
    )
    assert (
        score_annual_candidate(
            perfect_train_good_validation,
            score_mode="train_validation_100",
            field="rejection_reason",
        )
        == ""
    )
    assert (
        score_annual_candidate(
            perfect_train_bad_validation,
            score_mode="train_validation_100",
            field="rejection_reason",
        )
        == "validation_not_perfect"
    )


def test_crisis_stable_score_penalizes_large_false_negatives() -> None:
    good = {
        "feature_count": 3,
        "train_accuracy": 0.92,
        "train_negative_hits": 5,
        "train_negative_total": 6,
        "train_false_negative_big": 0,
        "train_return_mae": 0.1,
    }
    bad = {**good, "train_false_negative_big": 2}

    assert score_annual_candidate(good, score_mode="crisis_stable") > score_annual_candidate(
        bad,
        score_mode="crisis_stable",
    )


def test_crisis_stable_acceptance_requires_validation_stability_and_simple_rules() -> None:
    accepted = {
        "feature_count": 3,
        "train_accuracy": 0.92,
        "train_negative_hits": 5,
        "train_negative_total": 6,
        "train_false_negative_big": 0,
        "validation_accuracy": 0.91,
        "validation_negative_hits": 3,
        "validation_negative_total": 3,
        "validation_false_negative_big": 0,
        "always_positive_validation_accuracy": 0.73,
    }
    too_complex = {**accepted, "feature_count": 4}
    weak_validation = {**accepted, "validation_accuracy": 0.82}
    misses_stress = {**accepted, "validation_negative_hits": 1}

    assert score_annual_candidate(accepted, score_mode="crisis_stable", field="rejection_reason") == ""
    assert (
        score_annual_candidate(too_complex, score_mode="crisis_stable", field="rejection_reason")
        == "too_many_features"
    )
    assert (
        score_annual_candidate(weak_validation, score_mode="crisis_stable", field="rejection_reason")
        == "validation_accuracy"
    )
    assert (
        score_annual_candidate(misses_stress, score_mode="crisis_stable", field="rejection_reason")
        == "misses_validation_stress"
    )


def test_crisis_stable_catalog_can_exclude_fragile_feature_terms() -> None:
    examples = build_annual_examples(_daily_sample(), start_year=1981, end_year=2020)
    config = AnnualBeamConfig(
        stage=0,
        total_stages=2,
        seed_pool=40,
        beam_width=8,
        generations=1,
        mutations_per_parent=3,
        max_features=3,
        excluded_feature_terms=("santa",),
    )

    rows = run_annual_beam_search(examples, config)

    assert rows
    assert "santa" not in ";".join(str(row["specs"]).lower() for row in rows)


def test_annual_merge_counts_candidates_evaluated_from_stage_metadata() -> None:
    leaderboard = pd.DataFrame(
        [
            {"stage": 0, "stage_candidates_evaluated": 1000, "annual_score": 3.0, "accepted": False},
            {"stage": 0, "stage_candidates_evaluated": 1000, "annual_score": 2.0, "accepted": False},
            {"stage": 1, "stage_candidates_evaluated": 2000, "annual_score": 5.0, "accepted": True},
        ]
    )

    summary = summarize_annual_leaderboard(leaderboard)

    assert summary["rows"] == 3
    assert summary["candidates_evaluated"] == 3000
    assert summary["accepted"] == 1
