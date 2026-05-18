from pathlib import Path


def test_public_data_optimization_workflow_is_manual_and_uploads_artifacts() -> None:
    text = Path(".github/workflows/public-data-optimization.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "scripts/download_public_data.py" in text
    assert "actions/upload-artifact" in text
    assert "leaderboard.csv" in text


def test_survival_search_workflow_is_manual_and_never_mentions_locked_output() -> None:
    text = Path(".github/workflows/survival-search.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "scripts/run_survival_stage.py" in text
    assert "locked" not in text.lower()
    assert "--feature-panel" in text
    assert "survival_leaderboard.csv" in text
    assert "--total-stages 64" in text
    assert "actions/github-script" in text
    assert "Survival Search latest result" in text
