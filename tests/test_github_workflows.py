from pathlib import Path


def test_public_data_optimization_workflow_is_manual_and_uploads_artifacts() -> None:
    text = Path(".github/workflows/public-data-optimization.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "scripts/download_public_data.py" in text
    assert "actions/upload-artifact" in text
    assert "leaderboard.csv" in text


def test_weekly_sharpe_3methods_900_workflows_are_manual_balanced_and_mergeable() -> None:
    main = Path(".github/workflows/weekly-sharpe-3methods-max-parallel-900.yml").read_text(encoding="utf-8")
    block = Path(".github/workflows/weekly-sharpe-3methods-max-parallel-block.yml").read_text(encoding="utf-8")
    merge = Path(".github/workflows/weekly-sharpe-3methods-max-parallel-900-merge-now.yml").read_text(encoding="utf-8")
    config = Path("configs/weekly_sharpe_3methods_max_parallel_900.yaml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in main
    assert "push:" not in main
    assert "workflow_call" in block
    assert "source_run_id" in merge
    assert "block-a:" in main
    assert "block-b:" in main
    assert "block-c:" in main
    assert "block-d:" in main
    assert "stage_offset: 0" in main
    assert "stage_offset: 75" in main
    assert "stage_offset: 150" in main
    assert "stage_offset: 225" in main
    assert "max-parallel: 225" in block
    assert "method: [beam, genetic, machine_learning]" in block
    assert "--total-stages 300" in block
    assert "--time-budget-minutes" in block
    assert "weekly-sharpe-3methods-max-parallel-900-leaderboard" in main
    assert "weekly-sharpe-3methods-max-parallel-900-leaderboard" in merge
    assert "score_mode: train_sharpe_max_validation_80pct_report" in config
    assert "jobs_total: 900" in config
    assert "jobs_per_method: 300" in config
    assert "jobs_per_block: 225" in config
    assert "jobs_per_method_per_block: 75" in config
    assert "locked_opened: false" in config


def test_fair_ml_aurora_vs_github_30m_is_manual_paired_and_uploads_final_artifact() -> None:
    text = Path(".github/workflows/fair-ml-aurora-vs-github-30m.yml").read_text(encoding="utf-8")
    script = Path("scripts/fair_ml_aurora_vs_github.py").read_text(encoding="utf-8")
    readme = Path("runs pendientes/fair_ml_aurora_vs_github_30m/README.md").read_text(encoding="utf-8")

    assert "name: Fair ML Aurora vs GitHub 30m" in text
    assert "workflow_dispatch" in text
    assert "push:" in text
    assert ".github/fair-ml-aurora-vs-github-30m-trigger.txt" in text
    assert "scripts/fair_ml_aurora_vs_github.py" in text
    assert "${{ inputs.seconds_per_engine || '480' }}" in text
    assert "max-parallel: 2" in text
    assert "track: normalized" in text
    assert "order: aurora_first" in text
    assert "order: github_first" in text
    assert "repository: gomez5757/quantforge" not in text
    assert 'python -m pip install -e "quantforge[ml]"' not in text
    assert "python -m pip install numpy pandas pyyaml pydantic scikit-learn scipy pyarrow" in text
    assert "PYTHONPATH:" in text
    assert "fair_ml_aurora_vs_github.py run-pair" in text
    assert "fair_ml_aurora_vs_github.py merge" in text
    assert "fair-ml-aurora-vs-github-30m-results" in text
    assert "timeout-minutes: 25" in text
    assert "timeout-minutes: 5" in text
    assert "track: native" not in text

    assert "train_calmar" in script
    assert "validation_calmar >= 0.80 * train_calmar" in readme
    assert "locked_opened=false" in readme


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
    assert "wave:" in text
    assert "Independent search wave / seed bucket" in text
    assert "push:" not in text
    assert ".github/weekly-7methods-5h-trigger.txt" not in text
    assert "dependency-smoke:" in text
    assert "scripts/smoke_weekly_real_hpo_dependencies.py" in text
    assert 'python -m pip install -e ".[dev,hpo]"' in text
    assert "timeout-minutes: 420" in text
    assert "max-parallel: 35" in text
    assert "stage: [0, 1, 2" in text
    assert "method: [sobol_random_asha_real" not in text
    assert "methods=(sobol_random_asha_real optuna_tpe_hyperband dehb_real bohb_real smac_mf_real beam genetic)" in text
    assert 'for method in "${methods[@]}"; do' in text
    assert "budget_minutes=$((remaining_seconds / 7 / 60))" in text
    assert "--total-stages 35" in text
    assert '--wave "${{ inputs.wave }}"' in text
    assert "--time-budget-minutes \"$budget_minutes\"" in text
    assert "--file-prefix \"$FILE_PREFIX\"" in text
    assert "weekly-7methods-5h-fair-leaderboard" in text
    assert "weekly_7methods_5h_fair_leaderboard.csv" in text
    assert "state-dir" not in text
    assert "deadline_epoch=$(($(date +%s) + 25200))" in text
    assert text.index("deadline_epoch=$(($(date +%s) + 25200))") < text.index("actions/checkout@v4")
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
    assert "github_search_jobs: 35" in config
    assert "total_method_runs: 245" in config
    assert "total_search_jobs: 35" in config
    assert "time_budget_minutes: 50" in config
    assert "workflow_deadline_minutes: 420" in config
    assert "waves: 1" in config
    assert "stateful: false" in config
    assert "partial: false" in config
    assert "validation_role: report_only" in config
    assert "locked_opened: false" in config


def test_weekly_beam_genetic_2h_max_calmar_is_manual_balanced_and_locked_closed() -> None:
    text = Path(".github/workflows/weekly-beam-genetic-2h-max-calmar.yml").read_text(encoding="utf-8")
    config = Path("configs/weekly_beam_genetic_2h_max_calmar.yaml").read_text(encoding="utf-8")

    assert "name: Weekly Beam Genetic 2h Max Calmar" in text
    assert "workflow_dispatch" in text
    assert "push:" not in text
    assert "max-parallel: 200" in text
    assert "stage: [0, 1, 2" in text
    assert "method: [beam, genetic]" in text
    assert "--total-stages 100" in text
    assert "--time-budget-minutes 95" in text
    assert "--file-prefix \"$FILE_PREFIX\"" in text
    assert "scripts/audit_weekly_feature_catalog.py" in text
    assert "weekly-beam-genetic-2h-max-calmar-leaderboard" in text
    assert "weekly_beam_genetic_2h_max_calmar_leaderboard.csv" in text
    assert "weekly_beam_genetic_2h_feature_catalog.csv" in text
    assert "score_mode: train_calmar_max_validation_80pct_report" in config
    assert "methods:\n  - beam\n  - genetic" in config
    assert "stages_per_method: 100" in config
    assert "jobs_per_method: 100" in config
    assert "jobs_total: 200" in config
    assert "max_parallel: 200" in config
    assert "validation_role: report_only_for_score" in config
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


def test_weekly_7methods_overnight_search_is_manual_chunked_and_merge_free() -> None:
    text = Path(".github/workflows/weekly-7methods-overnight-search.yml").read_text(encoding="utf-8")
    wave = Path(".github/workflows/weekly-7methods-overnight-wave.yml").read_text(encoding="utf-8")
    config = Path("configs/weekly_7methods_overnight.yaml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "push:" not in text
    assert "waves:" in text
    assert "minutes_per_method_stage:" in text
    assert "max_parallel:" in text
    assert "weekly-7methods-overnight-public-panel" in text
    assert "wave-10:" in text
    assert "uses: ./.github/workflows/weekly-7methods-overnight-wave.yml" in text
    assert "Merge all methods" not in text
    assert "weekly-7methods-overnight-merged-leaderboard" not in text

    assert "workflow_call" in wave
    assert "method: [sobol_random_asha_real, optuna_tpe_hyperband, dehb_real, bohb_real, smac_mf_real, beam, genetic]" in wave
    assert "stage: [0, 1, 2" in wave
    assert "max-parallel: ${{ inputs.max_parallel }}" in wave
    assert "--method \"${{ matrix.method }}\"" in wave
    assert "--stage \"${{ matrix.stage }}\"" in wave
    assert "--time-budget-minutes \"${{ inputs.minutes_per_method_stage }}\"" in wave
    assert "weekly-7m-overnight-wave-${{ inputs.wave }}-${{ matrix.method }}-stage-${{ matrix.stage }}" in wave

    assert "waves: 10" in config
    assert "minutes_per_method_stage: 55" in config
    assert "stages_per_method: 35" in config
    assert "jobs_per_wave: 245" in config
    assert "locked_opened: false" in config
    assert "validation_role: report_only" in config


def test_weekly_7methods_overnight_merge_now_is_source_run_partial_merge() -> None:
    text = Path(".github/workflows/weekly-7methods-overnight-merge-now.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "source_run_id" in text
    assert "run-id: ${{ inputs.source_run_id }}" in text
    assert "continue-on-error: true" in text
    assert "weekly-7m-overnight-wave-*" in text
    assert "scripts/merge_weekly_7methods_overnight.py" in text
    assert "weekly-7methods-overnight-merged-leaderboard" in text
    assert "weekly_7methods_overnight_summary.json" in text


def test_weekly_7methods_overnight_stop_can_cancel_with_actions_write() -> None:
    text = Path(".github/workflows/weekly-7methods-overnight-stop.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "source_run_id" in text
    assert "actions: write" in text
    assert "/actions/runs/${SOURCE_RUN_ID}/cancel" in text
    assert "gh api -X POST" in text


def test_weekly_7methods_start_overnight_after_fair_is_github_native() -> None:
    text = Path(".github/workflows/weekly-7methods-start-overnight-after-fair.yml").read_text(encoding="utf-8")

    assert "workflow_run:" in text
    assert "Weekly Multi Asset SP500 Down 5pct 7 Methods 5h Fair Real HPO" in text
    assert "workflow_dispatch:" in text
    assert "actions: write" in text
    assert "concurrency:" in text
    assert "FAIR_WORKFLOW: weekly-7methods-5h-fair.yml" in text
    assert "OVERNIGHT_WORKFLOW: weekly-7methods-overnight-search.yml" in text
    assert "status=in_progress" in text
    assert "status=queued" in text
    assert "not starting duplicate" in text
    assert '"waves": "9"' in text
    assert '"minutes_per_method_stage": "55"' in text
    assert '"stages_per_method": "35"' in text
    assert '"max_parallel": "245"' in text
    assert "/dispatches" in text


def test_weekly_github_ml_until_first_valid_stops_after_found() -> None:
    text = Path(".github/workflows/weekly-github-ml-until-first-valid.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "validation_rule" in text
    assert "positive_calmar" in text
    assert "--method machine_learning" in text
    assert "check-wave-1" in text
    assert "search-wave-2:" in text
    assert "if: needs.check-wave-1.outputs.found != 'true'" in text
    assert "scripts/check_weekly_valid_candidate.py" in text
    assert "weekly-github-ml-until-first-valid-current-merge" in text
    assert "push:" not in text


def test_weekly_spy_sharpe_4methods_180_parallel_is_manual_balanced_and_locked_closed() -> None:
    text = Path(".github/workflows/weekly-spy-sharpe-4methods-2h-fair-180-parallel.yml").read_text(encoding="utf-8")
    config = Path("configs/weekly_spy_sharpe_4methods_180.yaml").read_text(encoding="utf-8")

    assert "name: Weekly SPY Sharpe 4 Methods 2h Fair 180 Parallel" in text
    assert "workflow_dispatch" in text
    assert "push:" not in text
    assert "max-parallel: 180" in text
    assert "stage: [0, 1, 2" in text
    assert "method: [beam, genetic, aurora_ml, github_ml]" in text
    assert "--total-stages 45" in text
    assert "--time-budget-minutes 100" in text
    assert "timeout-minutes: 125" in text
    assert "scripts/run_weekly_spy_sharpe_4methods_180_stage.py" in text
    assert "scripts/merge_weekly_spy_sharpe_4methods_180.py" in text
    assert "weekly-spy-sharpe-4methods-2h-fair-180-parallel-leaderboard" in text
    assert "weekly_spy_sharpe_4methods_180_parallelism.json" in text

    for method in ("beam", "genetic", "aurora_ml", "github_ml"):
        assert method in text
        assert f"  - {method}" in config
    assert "jobs_per_method: 45" in config
    assert "jobs_total: 180" in config
    assert "max_parallel: 180" in config
    assert "minutes_per_method_stage: 100" in config
    assert "timeout_minutes_per_job: 125" in config
    assert "score_mode: train_sharpe_max_validation_80pct_report" in config
    assert "validation_role: report_only_for_score" in config
    assert "locked_opened: false" in config


def test_weekly_spy_sharpe_10methods_9h_waves_is_pending_manual_balanced_and_locked_closed() -> None:
    text = Path(".github/workflows/weekly-spy-sharpe-10methods-9h-waves.yml").read_text(encoding="utf-8")
    wave = Path(".github/workflows/weekly-spy-sharpe-10methods-9h-wave.yml").read_text(encoding="utf-8")
    merge_now = Path(".github/workflows/weekly-spy-sharpe-10methods-9h-merge-now.yml").read_text(encoding="utf-8")
    stop = Path(".github/workflows/weekly-spy-sharpe-10methods-9h-stop.yml").read_text(encoding="utf-8")
    config = Path("configs/weekly_spy_sharpe_10methods_9h_waves.yaml").read_text(encoding="utf-8")
    pending = Path("runs pendientes/weekly_spy_sharpe_10methods_9h_waves/README.md").read_text(encoding="utf-8")

    assert "name: Weekly SPY Sharpe 10 Methods 9h Waves" in text
    assert "workflow_dispatch" in text
    assert "push:" not in text
    assert "wave-5:" in text
    assert "uses: ./.github/workflows/weekly-spy-sharpe-10methods-9h-wave.yml" in text
    assert "weekly-spy-sharpe-10methods-9h-waves-leaderboard" in text

    assert "workflow_call" in wave
    assert "stage: [0, 1, 2" in wave
    assert "method: [beam, genetic, sobol_random_asha_real, optuna_tpe_hyperband, dehb_real, bohb_real, smac_mf_real, bandit, aurora_ml, github_ml]" in wave
    assert "max-parallel: ${{ inputs.max_parallel }}" in wave
    assert "--state-dir state_in" in wave
    assert "scripts/run_weekly_spy_sharpe_10methods_9h_stage.py" in wave
    assert "scripts/merge_weekly_spy_sharpe_10methods_9h_state.py" in wave

    assert "source_run_id" in merge_now
    assert "run-id: ${{ inputs.source_run_id }}" in merge_now
    assert "continue-on-error: true" in merge_now
    assert "scripts/merge_weekly_spy_sharpe_10methods_9h.py" in merge_now
    assert "actions: write" in stop
    assert "/actions/runs/${SOURCE_RUN_ID}/cancel" in stop

    for method in ("beam", "genetic", "sobol_random_asha_real", "optuna_tpe_hyperband", "dehb_real", "bohb_real", "smac_mf_real", "bandit", "aurora_ml", "github_ml"):
        assert f"  - {method}" in config
    assert "waves: 5" in config
    assert "jobs_per_wave: 180" in config
    assert "jobs_per_method_total: 90" in config
    assert "jobs_total: 900" in config
    assert "max_parallel: 180" in config
    assert "minutes_per_method_stage: 85" in config
    assert "score_mode: train_sharpe_max_validation_80pct_report" in config
    assert "validation_role: report_only" in config
    assert "locked_opened: false" in config

    assert "Estado: `pendiente_no_lanzar`" in pending
    assert "No lanzar automaticamente" in pending


def test_weekly_multi_asset_sharpe_10methods_4h_waves_uses_feature_panel_and_locked_closed() -> None:
    text = Path(".github/workflows/weekly-multi-asset-sharpe-10methods-4h-waves.yml").read_text(encoding="utf-8")
    wave = Path(".github/workflows/weekly-multi-asset-sharpe-10methods-4h-wave.yml").read_text(encoding="utf-8")
    config = Path("configs/weekly_multi_asset_sharpe_10methods_4h_waves.yaml").read_text(encoding="utf-8")

    assert "name: Weekly Multi Asset Sharpe 10 Methods 4h Waves" in text
    assert "workflow_dispatch" in text
    assert "push:" in text
    assert ".github/weekly-multi-asset-sharpe-10methods-4h-trigger.txt" in text
    assert "inputs.minutes_per_method_stage || '110'" in text
    assert "inputs.max_parallel || '180'" in text
    assert "wave-2:" in text
    assert "wave-3:" not in text
    assert "download_public_data.py --feature-panel --output data/public/spy_daily.csv" in text
    assert "scripts/audit_weekly_multi_asset_panel.py" in text
    assert "weekly-multi-asset-sharpe-10methods-4h-waves-leaderboard" in text
    assert "--expected-jobs 360" in text

    assert "workflow_call" in wave
    assert "timeout-minutes: 120" in wave
    assert "method: [beam, genetic, sobol_random_asha_real, optuna_tpe_hyperband, dehb_real, bohb_real, smac_mf_real, bandit, aurora_ml, github_ml]" in wave
    assert "--total-stages 18" in wave
    assert "--file-prefix \"$FILE_PREFIX\"" in wave

    assert "waves: 2" in config
    assert "jobs_per_wave: 180" in config
    assert "jobs_per_method_total: 36" in config
    assert "jobs_total: 360" in config
    assert "max_parallel: 180" in config
    assert "minutes_per_method_stage: 110" in config
    assert "timeout_minutes_per_job: 120" in config
    assert "score_mode: train_sharpe_max_validation_80pct_report" in config
    assert "validation_role: report_only" in config
    assert "locked_opened: false" in config


def test_weekly_multi_asset_universal_robustness_runs_chunked_on_github() -> None:
    text = Path(".github/workflows/weekly-multi-asset-universal-robustness.yml").read_text(encoding="utf-8")

    assert "name: Weekly Multi Asset Universal Robustness" in text
    assert "workflow_dispatch" in text
    assert ".github/weekly-multi-asset-universal-robustness-trigger.txt" in text
    assert "run-id: ${{ env.SOURCE_RUN_ID }}" in text
    assert "weekly_multi_asset_sharpe_10methods_4h_verified.csv" in text
    assert "split_universal_robustness_leaderboard.py" in text
    assert "run_universal_strategy_robustness.py" in text
    assert "merge_universal_robustness_chunks.py" in text
    assert "max-parallel: 64" in text
    assert "--bootstrap-samples \"$BOOTSTRAP_SAMPLES\"" in text
    assert "weekly-multi-asset-universal-robustness-results" in text


def test_weekly_multi_asset_sharpe_positive_8methods_9h_500_shape() -> None:
    text = Path(".github/workflows/weekly-multi-asset-sharpe-positive-8methods-9h-500.yml").read_text(encoding="utf-8")
    block_a = Path(".github/workflows/weekly-multi-asset-sharpe-positive-8methods-9h-500-block-a.yml").read_text(encoding="utf-8")
    block_b = Path(".github/workflows/weekly-multi-asset-sharpe-positive-8methods-9h-500-block-b.yml").read_text(encoding="utf-8")
    config = Path("configs/weekly_multi_asset_sharpe_positive_8methods_9h_500.yaml").read_text(encoding="utf-8")

    assert "name: Weekly Multi Asset Sharpe Positive Years 8 Methods 9h 500" in text
    assert "workflow_dispatch" in text
    assert "download_public_data.py --feature-panel --output data/public/spy_daily.csv" in text
    assert "inputs.max_parallel || '500'" in text
    assert "--expected-jobs 992" in text
    assert "wave-1-a:" in text
    assert "wave-1-b:" in text
    assert "wave-2-a:" in text
    assert "wave-2-b:" in text
    assert "weekly-multi-asset-sharpe-positive-8methods-9h-500-leaderboard" in text

    assert "method: [dehb_real, genetic, beam, bandit]" in block_a
    assert "method: [github_ml, smac_mf_real, optuna_tpe_hyperband, sobol_random_asha_real]" in block_b
    assert "--total-stages 62" in block_a
    assert "--total-stages 62" in block_b
    assert "max-parallel: ${{ inputs.max_parallel }}" in block_a
    assert "max-parallel: ${{ inputs.max_parallel }}" in block_b
    assert "aurora_ml" not in block_a
    assert "aurora_ml" not in block_b
    assert "bohb_real" not in block_a
    assert "bohb_real" not in block_b

    assert "waves: 2" in config
    assert "jobs_total: 992" in config
    assert "jobs_per_wave: 496" in config
    assert "jobs_per_method_total: 124" in config
    assert "max_parallel: 500" in config
    assert "minutes_per_method_stage: 240" in config
    assert "score_mode: train_sharpe_positive_years_report_validation" in config
    assert "validation_role: report_only" in config
    assert "locked_opened: false" in config
    assert "aurora_ml" not in config
    assert "bohb_real" not in config


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
