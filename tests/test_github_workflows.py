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
    assert "push:" not in text
    assert "scripts/run_survival_stage.py" in text
    assert "locked" not in text.lower()
    assert "--feature-panel" in text
    assert "survival_leaderboard.csv" in text
    assert "--total-stages 64" in text
    assert "actions/github-script" in text
    assert "Survival Search latest result" in text


def test_survival_phase2_workflow_runs_combo_search_and_uploads_artifact() -> None:
    text = Path(".github/workflows/survival-phase2-combo.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "scripts/run_survival_combo_stage.py" in text
    assert "configs/survival_phase2_seeds.json" in text
    assert "survival-phase2-leaderboard" in text
    assert "--total-stages 64" in text
    assert "actions/github-script" in text
    assert "Survival Phase 2 latest result" in text


def test_survival_phase3_workflow_runs_walkforward_search_and_uploads_artifact() -> None:
    text = Path(".github/workflows/survival-phase3-walkforward.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "push:" not in text
    assert "scripts/run_survival_walkforward_stage.py" in text
    assert "configs/survival_phase3_github.yaml" in text
    assert "survival-phase3-leaderboard" in text
    assert "--total-stages 64" in text
    assert "Train block min Calmar" in text
    assert "Survival Phase 3 latest result" in text


def test_survival_phase4_workflow_runs_portfolio_regime_search_and_uploads_artifact() -> None:
    text = Path(".github/workflows/survival-phase4-portfolio.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "scripts/run_survival_portfolio_stage.py" in text
    assert "configs/survival_phase4_github.yaml" in text
    assert "survival-phase4-leaderboard" in text
    assert "--total-stages 32" in text
    assert "Survival Phase 4 latest result" in text


def test_survival_spy_only_workflow_enforces_single_asset_always_invested_search() -> None:
    text = Path(".github/workflows/survival-spy-only.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "push:" not in text
    assert "scripts/run_survival_spy_only_stage.py" in text
    assert "survival-spy-only-leaderboard" in text
    assert "--total-stages 64" in text
    assert "SPY-only" in text
    assert "TLT" not in text
    assert "GLD" not in text
    assert "SHY" not in text
    assert "CASH" not in text


def test_survival_spy_only_adaptive_workflow_runs_train_first_search() -> None:
    text = Path(".github/workflows/survival-spy-only-adaptive.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "scripts/run_survival_spy_only_adaptive_stage.py" in text
    assert "survival-spy-only-adaptive-leaderboard" in text
    assert "--total-stages 128" in text
    assert "--candidates-per-stage 2500" in text
    assert "SPY-only" in text
    assert "Train Calmar" in text
    assert "Cash allowed" in text
    assert "TLT" not in text
    assert "GLD" not in text
    assert "SHY" not in text
    assert "CASH" not in text


def test_survival_spy_only_beam_workflow_can_be_triggered_without_duplicates() -> None:
    text = Path(".github/workflows/survival-spy-only-beam.yml").read_text(encoding="utf-8")

    assert ".github/beam-trigger.txt" in text
    assert "workflow_run" not in text
    assert "workflow_dispatch" in text
    assert "scripts/run_survival_spy_only_beam_stage.py" in text
    assert "survival-spy-only-beam-leaderboard" in text
    assert "--total-stages 48" in text
    assert "--beam-width 24" in text
    assert "SPY-only" in text
    assert "TLT" not in text
    assert "GLD" not in text
    assert "SHY" not in text
    assert "CASH" not in text


def test_survival_spy_only_meta_workflow_adds_bayesian_bandit_and_genetic() -> None:
    text = Path(".github/workflows/survival-spy-only-meta.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "scripts/run_survival_spy_only_meta_stage.py" in text
    assert "method: [bayesian, bandit, genetic]" in text
    assert "--total-stages 16" in text
    assert "--candidates-per-stage 900" in text
    assert "survival-spy-only-meta-leaderboard" in text
    assert "SPY-only" in text
    assert "TLT" not in text
    assert "GLD" not in text
    assert "SHY" not in text
    assert "CASH" not in text


def test_survival_spy_only_shootout_workflow_compares_all_methods_manually() -> None:
    text = Path(".github/workflows/survival-spy-only-shootout.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "push:" not in text
    assert "method: [adaptive, beam, bayesian, bandit, genetic]" in text
    assert "scripts/run_survival_spy_only_shootout_stage.py" in text
    assert "scripts/merge_shootout_leaderboards.py" in text
    assert "--total-stages 12" in text
    assert "--budget 720" in text
    assert "survival-spy-only-shootout-leaderboard" in text
    assert "SPY-only" in text
    assert "TLT" not in text
    assert "GLD" not in text
    assert "SHY" not in text
    assert "CASH" not in text


def test_survival_spy_only_marathon_runs_five_methods_for_nine_hours() -> None:
    text = Path(".github/workflows/survival-spy-only-marathon-9h.yml").read_text(encoding="utf-8")

    assert ".github/marathon-trigger.txt" in text
    assert "workflow_dispatch" in text
    assert "method: [adaptive, beam, bayesian, bandit, genetic]" in text
    assert "lane: [0, 1, 2, 3]" in text
    assert "chunk: [0, 1, 2]" in text
    assert "--minutes 170" in text
    assert "scripts/run_survival_spy_only_marathon_chunk.py" in text
    assert "survival-spy-only-marathon-9h-leaderboard" in text
    assert "SPY-only" in text
    assert "TLT" not in text
    assert "GLD" not in text
    assert "SHY" not in text
    assert "CASH" not in text


def test_annual_sp500_beam_workflow_is_manual_and_keeps_locked_closed() -> None:
    text = Path(".github/workflows/annual-sp500-beam.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "push:" not in text
    assert "scripts/download_annual_public_data.py" in text
    assert "scripts/audit_annual_features.py" in text
    assert "scripts/run_annual_beam_search.py" in text
    assert "scripts/merge_annual_beam_leaderboards.py" in text
    assert "--total-stages 64" in text
    assert "annual_feature_coverage.csv" in text
    assert "annual-sp500-beam-leaderboard" in text
    assert "locked_opened: false" in text


def test_codespaces_devcontainer_uses_existing_python_image() -> None:
    text = Path(".devcontainer/devcontainer.json").read_text(encoding="utf-8")

    assert "mcr.microsoft.com/devcontainers/python:1-3.11-bookworm" in text


def test_survival_watchdog_runs_on_schedule_and_can_relaunch() -> None:
    text = Path(".github/workflows/survival-watchdog.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "actions: write" in text
    assert "issues: write" in text
    assert "scripts/watch_survival_search.py" in text
    assert "--relaunch-on-terminal-problem" in text
