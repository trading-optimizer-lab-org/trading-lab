from scripts.fair_ml_aurora_vs_github import _aggregate, _valid_candidate, _winner


def test_common_calmar_validity_rule_requires_ratio_cagr_and_closed_locked() -> None:
    valid = {
        "train_calmar": 2.0,
        "validation_calmar": 1.7,
        "train_cagr": 0.04,
        "validation_cagr": 0.03,
        "locked_opened": False,
    }

    assert _valid_candidate(valid, locked_opened=False)
    assert not _valid_candidate({**valid, "validation_calmar": 1.5}, locked_opened=False)
    assert not _valid_candidate({**valid, "validation_cagr": 0.02}, locked_opened=False)
    assert not _valid_candidate({**valid, "locked_opened": True}, locked_opened=False)
    assert not _valid_candidate(valid, locked_opened=True)


def test_normalized_winner_prefers_valid_per_min_then_validation_calmar() -> None:
    rows = _aggregate(
        [
            {
                "track": "normalized",
                "engine": "aurora",
                "effective_seconds": 480,
                "evaluated": 1000,
                "valid": 2,
                "best_valid": {"candidate_id": "a", "validation_calmar": 1.2, "train_calmar": 1.4},
            },
            {
                "track": "normalized",
                "engine": "github_ml",
                "effective_seconds": 480,
                "evaluated": 900,
                "valid": 2,
                "best_valid": {"candidate_id": "g", "validation_calmar": 1.5, "train_calmar": 1.6},
            },
        ]
    )

    assert _winner(rows)["engine"] == "github_ml"
