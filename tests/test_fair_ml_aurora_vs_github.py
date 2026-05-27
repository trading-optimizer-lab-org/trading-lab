from scripts.fair_ml_aurora_vs_github import _aggregate, _valid_candidate, _winner


def test_vendored_aurora_ml_search_is_importable() -> None:
    from aurora.research.ml_search import MLSearchConfig, run_ml_search

    assert MLSearchConfig(run_id="smoke").no_locked is True
    assert callable(run_ml_search)


def test_common_calmar_validity_rule_requires_validation_floor_cagr_and_closed_locked() -> None:
    valid = {
        "train_calmar": 0.01,
        "validation_calmar": 0.01,
        "locked_opened": False,
    }

    assert _valid_candidate(valid, locked_opened=False)
    assert not _valid_candidate({**valid, "validation_calmar": 0.0}, locked_opened=False)
    assert not _valid_candidate({**valid, "train_calmar": 0.0}, locked_opened=False)
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
