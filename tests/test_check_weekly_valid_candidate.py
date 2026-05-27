from pathlib import Path

from scripts.check_weekly_valid_candidate import main


def test_check_weekly_valid_candidate_outputs_found_for_positive_calmar(tmp_path: Path, monkeypatch) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "\n".join(
            [
                "candidate_id,method,train_calmar,validation_calmar,locked_opened",
                "bad,machine_learning,1.0,0.0,false",
                "ok,machine_learning,1.0,0.5,false",
            ]
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "check",
            "--leaderboard",
            str(leaderboard),
            "--summary-output",
            str(summary),
            "--rule",
            "positive_calmar",
        ],
    )

    assert main() == 0
    assert '"found": true' in summary.read_text(encoding="utf-8")
    assert '"valid_count": 1' in summary.read_text(encoding="utf-8")
