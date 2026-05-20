import pandas as pd

from scripts.merge_annual_train_only_verification import summarize_train_only_verification


def test_train_only_verification_counts_validation_only_after_merge() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "round_name": "r1",
                "stage": 0,
                "stage_candidates_evaluated": 10,
                "accepted": True,
                "train_accuracy": 1.0,
                "validation_accuracy": 1.0,
                "train_negative_hits": 2,
                "train_negative_total": 2,
                "validation_negative_hits": 1,
                "validation_negative_total": 1,
                "annual_score": 100.0,
            },
            {
                "candidate_id": "b",
                "round_name": "r2",
                "stage": 0,
                "stage_candidates_evaluated": 20,
                "accepted": True,
                "train_accuracy": 1.0,
                "validation_accuracy": 0.9,
                "train_negative_hits": 2,
                "train_negative_total": 2,
                "validation_negative_hits": 0,
                "validation_negative_total": 1,
                "annual_score": 90.0,
            },
        ]
    )
    verified = leaderboard[leaderboard["candidate_id"].eq("a")]

    summary = summarize_train_only_verification(leaderboard, verified)

    assert summary["candidates_evaluated"] == 30
    assert summary["accepted_train_perfect"] == 2
    assert summary["verified_train_validation_100"] == 1
    assert summary["score_mode"] == "train_only_100"
    assert summary["validation_role"] == "report_only"
    assert summary["locked_opened"] is False
