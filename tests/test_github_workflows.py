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
    assert "timeout-minutes: 60" in text
    assert "if: always()" in text
    assert "annual_feature_coverage.csv" in text
    assert "annual-sp500-beam-leaderboard" in text
    assert "locked_opened: false" in text


def test_annual_sp500_train_only_100_workflow_optimizes_train_only() -> None:
    text = Path(".github/workflows/annual-sp500-train-only-100.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "push:" in text
    assert ".github/train-only-100-trigger.txt" in text
    assert "configs/annual_sp500_train_only_100.yaml" in text
    assert "--score-mode train_only_100" in text
    assert "--max-features 5" in text
    assert "--total-stages 64" in text
    assert "annual-sp500-train-only-100-leaderboard" in text
    assert "locked_opened: false" in text


def test_annual_sp500_crisis_stable_workflow_runs_simple_stable_detector() -> None:
    text = Path(".github/workflows/annual-sp500-crisis-stable.yml").read_text(encoding="utf-8")
    config = Path("configs/annual_sp500_crisis_stable.yaml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "push:" in text
    assert ".github/crisis-stable-trigger.txt" in text
    assert "configs/annual_sp500_crisis_stable.yaml" in text
    assert "--score-mode crisis_stable" in text
    assert "--max-features 3" in text
    assert "--total-stages 64" in text
    assert "--seed-pool 2500" in text
    assert "--beam-width 96" in text
    assert "--generations 16" in text
    assert "--mutations-per-parent 24" in text
    assert "--exclude-feature-contains santa" in text
    assert "annual-sp500-crisis-stable-leaderboard" in text
    assert "score_mode: crisis_stable" in config
    assert "max_features: 3" in config
    assert "santa" in config


def test_annual_validation_optimization_workflows_are_removed() -> None:
    forbidden_paths = (
        ".github/workflows/annual-sp500-train-validation-100.yml",
        ".github/workflows/annual-sp500-train-validation-wide.yml",
        ".github/workflows/annual-sp500-loop-until-goal.yml",
        ".github/train-validation-100-trigger.txt",
        ".github/train-validation-wide-trigger.txt",
        ".github/annual-loop-trigger.txt",
        "configs/annual_sp500_train_validation_100.yaml",
        "configs/annual_sp500_train_validation_100_wide.yaml",
        "configs/annual_sp500_loop_until_goal.yaml",
    )

    for path in forbidden_paths:
        assert not Path(path).exists()


def test_annual_validation_optimization_score_modes_are_not_runnable() -> None:
    script = Path("scripts/run_annual_beam_search.py").read_text(encoding="utf-8")

    assert "FORBIDDEN_VALIDATION_OPTIMIZATION_SCORE_MODES" in script
    assert "train_validation_100" in script
    assert "train_validation_hunt_100" in script
    assert "validation cannot be used as an optimization target" in script


def test_annual_sp500_train_only_verify_loop_keeps_validation_report_only() -> None:
    text = Path(".github/workflows/annual-sp500-train-only-verify-loop.yml").read_text(encoding="utf-8")
    config = Path("configs/annual_sp500_train_only_verify_loop.yaml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "push:" in text
    assert ".github/train-only-verify-trigger.txt" in text
    assert "round: [all_features_5, all_features_8, no_santa_8, no_santa_no_vix_russell_8]" in text
    assert "--score-mode train_only_100" in text
    assert "train_validation_hunt_100" not in text
    assert "--score-mode train_validation_100" not in text
    assert "scripts/merge_annual_train_only_verification.py" in text
    assert "validation_role: report_only" in text
    assert "locked_opened: false" in text
    assert "annual-sp500-train-only-verify-leaderboard" in text
    assert "score_mode: train_only_100" in config
    assert "max_features: 8" in config


def test_heavy_workflows_do_not_run_on_push() -> None:
    for path in (
        ".github/workflows/annual-sp500-beam.yml",
        ".github/workflows/survival-spy-only-adaptive.yml",
        ".github/workflows/codespaces-smoke.yml",
    ):
        text = Path(path).read_text(encoding="utf-8")
        assert "workflow_dispatch" in text
        assert "push:" not in text


def test_weekly_7methods_12h_stateful_workflow_is_manual_balanced_and_locked_closed() -> None:
    text = Path(".github/workflows/weekly-7methods-12h-stateful.yml").read_text(encoding="utf-8")
    config = Path("configs/weekly_7methods_12h_stateful.yaml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "push:" in text
    assert ".github/weekly-7methods-12h-trigger.txt" in text
    assert "max-parallel: 245" in text
    assert text.count("--time-budget-minutes 240") == 6
    assert text.count("--expected-files-per-method 70") == 3
    assert text.count("--allow-missing-files-per-method 1") == 3
    assert "continue-on-error: true" in text
    assert "weekly-7methods-12h-stateful-sp500-down-5pct-leaderboard" in text
    assert "weekly-7methods-public-panel" in text
    for method in (
        "sobol_random_asha",
        "tpe_asha_lite",
        "dehb_lite",
        "bohb_lite",
        "smac_mf_lite",
        "beam",
        "genetic",
    ):
        assert method in text
        assert f"  - {method}" in config
    assert "jobs_per_method_per_wave: 70" in config
    assert "waves: 3" in config
    assert "validation_role: report_only" in config
    assert "locked_opened: false" in config


def test_weekly_7methods_12h_merge_only_recovers_method_merges_from_source_run() -> None:
    text = Path(".github/workflows/weekly-7methods-12h-merge-only.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert ".github/weekly-7methods-merge-only-trigger.txt" in text
    assert "source_run_id" in text
    assert "weekly-7methods-12h-method-merge-*" in text
    assert "run-id: ${{ inputs.source_run_id || env.SOURCE_RUN_ID }}" in text
    assert "weekly-7methods-12h-wave1-recovered-leaderboard" in text


def test_weekly_7methods_5h_fair_workflow_runs_one_balanced_wave_from_zero() -> None:
    text = Path(".github/workflows/weekly-7methods-5h-fair.yml").read_text(encoding="utf-8")
    config = Path("configs/weekly_7methods_5h_fair.yaml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "push:" not in text
    assert ".github/weekly-7methods-5h-trigger.txt" not in text
    assert "dependency-smoke:" in text
    assert "scripts/smoke_weekly_real_hpo_dependencies.py" in text
    assert 'python -m pip install -e ".[dev,hpo]"' in text
    assert "max-parallel: 245" in text
    assert text.index("stage: [0, 1, 2") < text.index("method: [sobol_random_asha_real")
    assert "--total-stages 35" in text
    assert "--time-budget-minutes \"$budget_minutes\"" in text
    assert "--file-prefix \"$FILE_PREFIX\"" in text
    assert "weekly-7methods-5h-fair-leaderboard" in text
    assert "weekly_7methods_5h_fair_leaderboard.csv" in text
    assert "state-dir" not in text
    assert "deadline_epoch=$(($(date +%s) + 18000))" in text
    assert text.index("deadline_epoch=$(($(date +%s) + 18000))") < text.index("actions/checkout@v4")
    assert "deadline=\"${{ needs.dependency-smoke.outputs.deadline_epoch }}\"" in text
    assert "needs: [dependency-smoke, data]" in text
    assert "needs: [dependency-smoke, data, search]" in text
    for method in (
        "sobol_random_asha_real",
        "optuna_tpe_hyperband",
        "dehb_real",
        "bohb_real",
        "smac_mf_real",
        "beam",
        "genetic",
    ):
        assert method in text
        assert f"  - {method}" in config
    assert "jobs_per_method: 35" in config
    assert "total_search_jobs: 245" in config
    assert "waves: 1" in config
    assert "stateful: false" in config
    assert "partial: false" in config
    assert "validation_role: report_only" in config
    assert "locked_opened: false" in config


def test_github_actions_unblock_smoke_is_manual_single_checkout_job() -> None:
    text = Path(".github/workflows/github-actions-unblock-smoke.yml").read_text(encoding="utf-8")

    assert "name: GitHub Actions Unblock Smoke" in text
    assert "workflow_dispatch" in text
    assert "push:" not in text
    assert "matrix:" not in text
    assert "actions/checkout@v4" in text
    assert "echo checkout_ok" in text
    assert "timeout-minutes: 10" in text
    assert "upload-artifact" not in text
    assert "pip install" not in text


def test_ci_is_manual_or_pull_request_only() -> None:
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "pull_request:" in text
    assert "push:" not in text


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
