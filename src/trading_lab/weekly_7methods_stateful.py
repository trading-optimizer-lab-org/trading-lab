from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from trading_lab.monthly_risk import MonthlyRiskSearchConfig, _json_safe
from trading_lab.weekly_multi_asset import (
    WEEKLY_MAX_SHARPE_SCORE_MODE,
    WEEKLY_ASSET_SELECTORS,
    WeeklyMachineLearningCandidate,
    WeeklyMultiAssetCandidate,
    _asset_universes,
    _available_assets_from_examples,
    _build_weekly_spec_catalog,
    _candidate_from_row,
    _candidate_id,
    _catalog_groups,
    _choose_group,
    _crossover,
    _evaluate_unique,
    _group_rewards,
    _method_summary,
    _mutate_candidate,
    _random_candidate,
    _select_beam,
    _spec_weights_from_rows,
    _verified,
    _weighted_candidate,
    _ml_feature_names,
    evaluate_weekly_machine_learning_candidate,
    evaluate_weekly_multi_asset_candidate,
    merge_weekly_multi_asset_leaderboards,
)


STATEFUL_WEEKLY_METHODS = (
    "sobol_random_asha",
    "tpe_asha_lite",
    "dehb_lite",
    "bohb_lite",
    "smac_mf_lite",
    "beam",
    "genetic",
)

FAIR_5H_WEEKLY_METHODS = (
    "sobol_random_asha_real",
    "optuna_tpe_hyperband",
    "dehb_real",
    "bohb_real",
    "smac_mf_real",
    "beam",
    "genetic",
)

SHARPE_3METHOD_WEEKLY_METHODS = ("beam", "genetic", "machine_learning")
SHARPE_10METHOD_WEEKLY_METHODS = (
    "beam",
    "genetic",
    "sobol_random_asha_real",
    "optuna_tpe_hyperband",
    "dehb_real",
    "bohb_real",
    "smac_mf_real",
    "bandit",
    "machine_learning",
)

ALL_STATEFUL_WEEKLY_METHODS = tuple(
    dict.fromkeys((*STATEFUL_WEEKLY_METHODS, *FAIR_5H_WEEKLY_METHODS, *SHARPE_3METHOD_WEEKLY_METHODS, *SHARPE_10METHOD_WEEKLY_METHODS))
)

METHOD_ENGINE_ALIASES = {
    "sobol_random_asha_real": "real_hpo",
    "optuna_tpe_hyperband": "real_hpo",
    "dehb_real": "real_hpo",
    "bohb_real": "real_hpo",
    "smac_mf_real": "real_hpo",
}

EXPECTED_TRAIN_YEARS = 14
EXPECTED_VALIDATION_YEARS = 12
EXPECTED_TRAIN_DOWN_YEARS = 3
EXPECTED_VALIDATION_DOWN_YEARS = 2

STATE_TOP_CANDIDATES = 500


def run_stateful_weekly_search(
    examples: pd.DataFrame,
    config: MonthlyRiskSearchConfig,
    *,
    method: str,
    wave: int,
    stage: int,
    time_budget_minutes: float,
    state_dir: str | Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if method not in ALL_STATEFUL_WEEKLY_METHODS:
        raise ValueError(f"unknown stateful weekly method: {method}")
    catalog = _build_weekly_spec_catalog(examples)
    assets = _available_assets_from_examples(examples)
    if not catalog or not assets:
        return [], _empty_state(method, wave, stage)

    engine_method = _engine_method(method)
    prior_state = load_method_state(state_dir, method)
    prior_candidates = _state_candidates(prior_state)
    if engine_method == "machine_learning":
        return run_weekly_machine_learning_search(
            examples,
            config,
            method=method,
            wave=wave,
            stage=stage,
            time_budget_minutes=time_budget_minutes,
        )
    if engine_method == "real_hpo":
        return run_real_hpo_weekly_search(
            examples,
            config,
            method=method,
            wave=wave,
            stage=stage,
            time_budget_minutes=time_budget_minutes,
            prior_candidates=prior_candidates,
        )

    seed = int(config.random_seed + wave * 1_000_000 + stage * 10_000 + _stateful_method_offset(method))
    rng = np.random.default_rng(seed)

    seen: set[tuple[tuple[str, ...], tuple[str, ...], str, float, float, float]] = set()
    rows: list[dict[str, object]] = []
    start = time.monotonic()
    deadline = start + max(1.0, float(time_budget_minutes) * 60.0)
    iteration = 0

    if prior_candidates:
        rows.extend(
            _evaluate_and_stamp(
                examples,
                prior_candidates,
                seen,
                method=method,
                wave=wave,
                stage=stage,
                start=start,
                score_mode=config.score_mode,
            )
        )

    while time.monotonic() < deadline or iteration == 0:
        candidates = _next_candidates(engine_method, catalog, assets, config=config, rng=rng, rows=rows, prior=prior_candidates, iteration=iteration)
        if not candidates:
            break
        child_rows = _evaluate_and_stamp(
            examples,
            candidates,
            seen,
            method=method,
            wave=wave,
            stage=stage,
            start=start,
            score_mode=config.score_mode,
        )
        rows.extend(child_rows)
        iteration += 1
        if time_budget_minutes <= 0:
            break

    rows = _trim_stage_rows(rows, config.top_rows_per_stage)
    state = build_method_state(rows, method=method, wave=wave, stage=stage)
    return rows, state


def run_weekly_machine_learning_search(
    examples: pd.DataFrame,
    config: MonthlyRiskSearchConfig,
    *,
    method: str,
    wave: int,
    stage: int,
    time_budget_minutes: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    assets = _available_assets_from_examples(examples)
    feature_names = _ml_feature_names(examples)
    if not assets or not feature_names:
        return [], _empty_state(method, wave, stage)
    stage_features = [name for index, name in enumerate(feature_names) if index % config.total_stages == config.stage] or feature_names
    universes = _asset_universes(assets)
    seed = int(config.random_seed + wave * 1_000_000 + stage * 10_000 + _stateful_method_offset(method))
    rng = np.random.default_rng(seed)
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str, str, float, int, int, float, float, float]] = set()
    rows: list[dict[str, object]] = []
    start = time.monotonic()
    deadline = start + max(1.0, float(time_budget_minutes) * 60.0)
    iteration = 0
    while time.monotonic() < deadline or iteration == 0:
        candidates = _next_ml_candidates(stage_features, universes, config=config, rng=rng, stage=stage, iteration=iteration)
        if not candidates:
            break
        for candidate in candidates:
            key = (
                tuple(sorted(candidate.features)),
                tuple(sorted(candidate.assets)),
                candidate.selector,
                candidate.model,
                round(float(candidate.alpha), 4),
                int(candidate.n_estimators),
                int(candidate.max_depth),
                round(float(candidate.learning_rate), 4),
                round(float(candidate.scale), 4),
                round(float(candidate.intercept), 4),
            )
            if key in seen:
                continue
            seen.add(key)
            row, _, _ = evaluate_weekly_machine_learning_candidate(
                examples,
                candidate,
                method=method,
                score_mode=config.score_mode or WEEKLY_MAX_SHARPE_SCORE_MODE,
            )
            elapsed = max(0.0, time.monotonic() - start)
            row["method"] = method
            row["wave"] = int(wave)
            row["stage"] = int(stage)
            row["elapsed_seconds"] = float(elapsed)
            row["candidates_evaluated"] = int(len(rows) + 1)
            row["first_seen_wave"] = int(wave)
            row["first_seen_minute"] = float(elapsed / 60.0)
            row["accepted"] = bool(row.get("accepted")) and _has_expected_train_counts(row)
            row["verified_sharpe_robust"] = (
                bool(row.get("accepted"))
                and _has_expected_validation_counts(row)
                and bool(row.get("verified_sharpe_robust"))
            )
            row["verified_train_validation_5pct"] = False
            row["locked_opened"] = False
            row["validation_role"] = "report_only"
            rows.append(row)
        iteration += 1
        if time_budget_minutes <= 0:
            break
    rows = _trim_stage_rows(rows, config.top_rows_per_stage)
    state = build_method_state(rows, method=method, wave=wave, stage=stage)
    return rows, state


def run_real_hpo_weekly_search(
    examples: pd.DataFrame,
    config: MonthlyRiskSearchConfig,
    *,
    method: str,
    wave: int,
    stage: int,
    time_budget_minutes: float,
    prior_candidates: list[WeeklyMultiAssetCandidate] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    catalog = _build_weekly_spec_catalog(examples)
    assets = _available_assets_from_examples(examples)
    if not catalog or not assets:
        return [], _empty_state(method, wave, stage)
    stage_catalog = [spec for index, spec in enumerate(catalog) if index % config.total_stages == config.stage] or catalog
    universes = _asset_universes(assets)
    rng = np.random.default_rng(config.random_seed + wave * 1_000_000 + stage * 10_000 + _stateful_method_offset(method))
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str, float, float, float]] = set()
    rows: list[dict[str, object]] = []
    start = time.monotonic()
    deadline = start + max(1.0, float(time_budget_minutes) * 60.0)
    if prior_candidates:
        rows.extend(
            _evaluate_and_stamp(
                examples,
                prior_candidates,
                seen,
                method=method,
                wave=wave,
                stage=stage,
                start=start,
                score_mode=config.score_mode,
            )
        )

    def evaluate(candidate: WeeklyMultiAssetCandidate, *, backend: str) -> float:
        stamped = _evaluate_and_stamp(
            examples,
            [candidate],
            seen,
            method=method,
            wave=wave,
            stage=stage,
            start=start,
            score_mode=config.score_mode,
        )
        for row in stamped:
            row["hpo_backend"] = backend
            row["method_real"] = True
        rows.extend(stamped)
        if not stamped:
            return -1_000_000.0
        return float(stamped[-1].get("weekly_multi_asset_score", -1_000_000.0) or -1_000_000.0)

    if method == "sobol_random_asha_real":
        _run_optuna_real(stage_catalog, universes, config=config, deadline=deadline, rng=rng, evaluate=evaluate, sampler_name="qmc", pruner_name="asha")
    elif method == "optuna_tpe_hyperband":
        _run_optuna_real(stage_catalog, universes, config=config, deadline=deadline, rng=rng, evaluate=evaluate, sampler_name="tpe", pruner_name="hyperband")
    elif method == "dehb_real":
        _run_dehb_real(stage_catalog, universes, config=config, deadline=deadline, evaluate=evaluate)
    elif method == "bohb_real":
        _run_bohb_real(stage_catalog, universes, config=config, deadline=deadline, evaluate=evaluate, run_id=f"bohb-{wave}-{stage}-{int(start)}")
    elif method == "smac_mf_real":
        _run_smac_real(stage_catalog, universes, config=config, deadline=deadline, evaluate=evaluate, seed=int(rng.integers(1, 2_000_000_000)))
    else:
        raise ValueError(f"unknown real HPO method: {method}")

    rows = _trim_stage_rows(rows, config.top_rows_per_stage)
    state = build_method_state(rows, method=method, wave=wave, stage=stage)
    return rows, state


def write_stateful_weekly_outputs(
    rows: list[dict[str, object]],
    state: dict[str, object],
    output_dir: str | Path,
    *,
    method: str,
    wave: int,
    stage: int,
    file_prefix: str = "weekly_7methods_12h",
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.DataFrame(rows)
    leaderboard.to_csv(output / f"{file_prefix}_leaderboard.csv", index=False)
    leaderboard.to_csv(output / f"{file_prefix}_leaderboard_stage_{method}_{wave}_{stage}.csv", index=False)
    verified = _strict_verified(leaderboard)
    verified.to_csv(output / f"{file_prefix}_verified.csv", index=False)
    state_dir = output / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{file_prefix}_state_{method}_{wave}_{stage}.json"
    state_path.write_text(json.dumps(_json_safe(state), indent=2, sort_keys=True), encoding="utf-8")
    summary = _stage_summary(rows, method=method, wave=wave, stage=stage)
    (output / f"{file_prefix}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def merge_state_files(
    paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    state_top: int = STATE_TOP_CANDIDATES,
    expected_files_per_method: int = 0,
    allow_missing_files_per_method: int = 0,
) -> dict[str, object]:
    output = Path(output_dir)
    state_output = output / "state"
    state_output.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, object]]] = {method: [] for method in STATEFUL_WEEKLY_METHODS}
    state_files_by_method: dict[str, int] = {method: 0 for method in STATEFUL_WEEKLY_METHODS}
    raw_states = 0
    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            continue
        raw_states += 1
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        method = str(data.get("method", ""))
        if method in grouped:
            state_files_by_method[method] += 1
            grouped[method].extend(list(data.get("candidates", [])))

    if expected_files_per_method > 0:
        allowed_minimum = max(0, expected_files_per_method - max(0, allow_missing_files_per_method))
        missing = {
            method: count
            for method, count in state_files_by_method.items()
            if count < allowed_minimum
        }
        if missing:
            raise ValueError(
                "missing weekly 7-method state files: "
                + ", ".join(f"{method}={count}/{expected_files_per_method}" for method, count in sorted(missing.items()))
            )
    missing_state_files_by_method = {
        method: max(0, int(expected_files_per_method) - int(count))
        for method, count in state_files_by_method.items()
        if expected_files_per_method > 0 and count < expected_files_per_method
    }

    summary_methods: list[dict[str, object]] = []
    for method, candidates in grouped.items():
        dedup = _dedupe_state_candidates(candidates)
        kept = [
            _sanitize_state_candidate(row)
            for row in sorted(dedup, key=lambda item: float(item.get("train_score", 0.0) or 0.0), reverse=True)[:state_top]
        ]
        state = {
            "method": method,
            "candidates": kept,
            "candidate_count": len(kept),
            "source_state_count": raw_states,
            "validation_role": "report_only",
            "locked_opened": False,
        }
        (state_output / f"{method}.json").write_text(json.dumps(_json_safe(state), indent=2, sort_keys=True), encoding="utf-8")
        summary_methods.append({"method": method, "state_candidates": len(kept), "raw_candidates": len(candidates)})
    summary = {
        "state_files": raw_states,
        "state_files_by_method": state_files_by_method,
        "expected_files_per_method": int(expected_files_per_method),
        "allow_missing_files_per_method": int(max(0, allow_missing_files_per_method)),
        "missing_state_files_by_method": missing_state_files_by_method,
        "methods": summary_methods,
        "validation_role": "report_only",
        "locked_opened": False,
    }
    (output / "weekly_7methods_12h_state_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def merge_stateful_weekly_leaderboards(
    paths: list[str | Path],
    output_dir: str | Path,
    *,
    examples: pd.DataFrame | None = None,
    max_output_rows: int = 50_000,
    file_prefix: str = "weekly_7methods_12h",
    expected_methods: Iterable[str] | None = None,
) -> dict[str, object]:
    temp = Path(output_dir) / "_weekly_multi_temp"
    summary = merge_weekly_multi_asset_leaderboards(paths, temp, examples=examples, progress_every=25, max_output_rows=max_output_rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    name_map = {
        "weekly_multi_asset_sp500_down_5pct_leaderboard.csv": f"{file_prefix}_leaderboard.csv",
        "weekly_multi_asset_sp500_down_5pct_verified.csv": f"{file_prefix}_verified.csv",
        "weekly_multi_asset_sp500_down_5pct_methods.csv": f"{file_prefix}_methods.csv",
        "weekly_multi_asset_sp500_down_5pct_summary.json": f"{file_prefix}_summary.json",
        "weekly_multi_asset_sp500_down_5pct_year_by_year.csv": f"{file_prefix}_year_by_year.csv",
        "weekly_multi_asset_sp500_down_5pct_weekly_positions.csv": f"{file_prefix}_weekly_positions.csv",
    }
    for source, target in name_map.items():
        source_path = temp / source
        if source_path.exists():
            (output / target).write_bytes(source_path.read_bytes())

    leaderboard_path = output / f"{file_prefix}_leaderboard.csv"
    if leaderboard_path.exists():
        leaderboard = pd.read_csv(leaderboard_path)
    else:
        leaderboard = pd.DataFrame()
    verified = _strict_verified(leaderboard)
    verified.to_csv(output / f"{file_prefix}_verified.csv", index=False)
    method_summary = _stateful_method_summary(leaderboard, verified)
    method_summary.to_csv(output / f"{file_prefix}_methods.csv", index=False)
    efficiency = _stateful_efficiency(leaderboard, verified)
    efficiency.to_csv(output / f"{file_prefix}_efficiency.csv", index=False)
    method_list = list(expected_methods if expected_methods is not None else STATEFUL_WEEKLY_METHODS)
    score_mode = str(leaderboard.iloc[0].get("score_mode", summary.get("score_mode", ""))) if not leaderboard.empty else str(summary.get("score_mode", ""))
    verified_down_5pct = int(leaderboard.get("verified_train_validation_5pct", pd.Series(dtype=bool)).astype(bool).sum()) if not leaderboard.empty else 0
    verified_calmar = int(leaderboard.get("verified_calmar_similarity", pd.Series(dtype=bool)).astype(bool).sum()) if not leaderboard.empty else 0
    verified_sharpe = int(leaderboard.get("verified_sharpe_robust", pd.Series(dtype=bool)).astype(bool).sum()) if not leaderboard.empty else 0

    final_summary = {
        **summary,
        "rows": int(len(leaderboard)),
        "verified_train_validation_5pct": verified_down_5pct,
        "verified_calmar_similarity": verified_calmar,
        "verified_sharpe_robust": verified_sharpe,
        "unique_verified_train_validation_5pct": int(
            leaderboard.loc[leaderboard.get("verified_train_validation_5pct", pd.Series(dtype=bool)).astype(bool), "candidate_id"].nunique()
        ) if "candidate_id" in leaderboard and "verified_train_validation_5pct" in leaderboard else 0,
        "unique_verified_calmar_similarity": int(
            leaderboard.loc[leaderboard.get("verified_calmar_similarity", pd.Series(dtype=bool)).astype(bool), "candidate_id"].nunique()
        ) if "candidate_id" in leaderboard and "verified_calmar_similarity" in leaderboard else 0,
        "unique_verified_sharpe_robust": int(
            leaderboard.loc[leaderboard.get("verified_sharpe_robust", pd.Series(dtype=bool)).astype(bool), "candidate_id"].nunique()
        ) if "candidate_id" in leaderboard and "verified_sharpe_robust" in leaderboard else 0,
        "best_candidate": str(leaderboard.iloc[0]["candidate_id"]) if not leaderboard.empty else None,
        "best_method": str(leaderboard.iloc[0].get("method", "")) if not leaderboard.empty else None,
        "best_train_score": float(leaderboard.iloc[0].get("weekly_multi_asset_score", 0.0)) if not leaderboard.empty else None,
        "best_train_calmar": float(leaderboard.iloc[0].get("train_calmar", 0.0)) if not leaderboard.empty else None,
        "best_validation_calmar": float(leaderboard.iloc[0].get("validation_calmar", 0.0)) if not leaderboard.empty else None,
        "best_train_sharpe": float(leaderboard.iloc[0].get("train_sharpe", 0.0)) if not leaderboard.empty else None,
        "best_validation_sharpe": float(leaderboard.iloc[0].get("validation_sharpe", 0.0)) if not leaderboard.empty else None,
        "best_verified_candidate": str(verified.iloc[0]["candidate_id"]) if not verified.empty else None,
        "jobs_started": _unique_stage_count(leaderboard),
        "jobs_completed": _unique_stage_count(leaderboard),
        "jobs_failed": _failed_stage_count(leaderboard),
        "partial": False,
        "locked_opened": False,
        "validation_role": "report_only",
        "score_mode": score_mode,
        "methods": method_list,
        "expected_train_years": EXPECTED_TRAIN_YEARS,
        "expected_validation_years": EXPECTED_VALIDATION_YEARS,
        "expected_train_down_years": EXPECTED_TRAIN_DOWN_YEARS,
        "expected_validation_down_years": EXPECTED_VALIDATION_DOWN_YEARS,
    }
    (output / f"{file_prefix}_summary.json").write_text(json.dumps(_json_safe(final_summary), indent=2, sort_keys=True), encoding="utf-8")
    return final_summary


def load_method_state(state_dir: str | Path | None, method: str) -> dict[str, object]:
    if not state_dir:
        return {}
    root = Path(state_dir)
    for candidate in (root / "state" / f"{method}.json", root / f"{method}.json"):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    matches = sorted(root.rglob(f"{method}.json"))
    if matches:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    return {}


def build_method_state(rows: list[dict[str, object]], *, method: str, wave: int, stage: int) -> dict[str, object]:
    candidates = []
    for row in _select_beam(rows, STATE_TOP_CANDIDATES):
        candidates.append(_state_candidate_from_row(row))
    return {
        "method": method,
        "wave": wave,
        "stage": stage,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "validation_role": "report_only",
        "locked_opened": False,
    }


def _next_candidates(
    method: str,
    catalog: list[str],
    assets: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
    rows: list[dict[str, object]],
    prior: list[WeeklyMultiAssetCandidate],
    iteration: int,
) -> list[WeeklyMultiAssetCandidate]:
    batch_size = min(
        max(32, min(512, config.beam_width * max(1, config.mutations_per_parent // 4))),
        max(1, int(config.seed_pool)),
    )
    if iteration == 0 and not rows:
        return _seed_candidates_lite(catalog, assets, config=config, rng=rng, method=method, limit=batch_size)
    if method == "beam":
        return _beam_candidates(catalog, assets, rows, config=config, rng=rng, batch_size=batch_size)
    if method == "genetic":
        return _genetic_candidates(catalog, assets, rows, prior, config=config, rng=rng, batch_size=batch_size)
    if method == "tpe_asha_lite":
        weights = _spec_weights_from_rows(rows, catalog)
        return [_weighted_candidate(catalog, weights, assets, config=config, rng=rng) for _ in range(batch_size)]
    if method == "dehb_lite":
        return _dehb_candidates(catalog, assets, rows, prior, config=config, rng=rng, batch_size=batch_size)
    if method == "bohb_lite":
        weights = _spec_weights_from_rows(rows, catalog)
        return [
            _weighted_candidate(catalog, weights, assets, config=config, rng=rng) if index % 2 else _random_candidate(catalog, assets, config=config, rng=rng)
            for index in range(batch_size)
        ]
    if method == "smac_mf_lite":
        groups = _catalog_groups(catalog)
        rewards = _group_rewards(rows, groups)
        return [_random_candidate(groups.get(_choose_group(rewards, rng), catalog) or catalog, assets, config=config, rng=rng) for _ in range(batch_size)]
    if method == "bandit":
        groups = _catalog_groups(catalog)
        rewards = _group_rewards(rows, groups)
        return [
            _random_candidate(groups.get(_choose_group(rewards, rng), catalog) or catalog, assets, config=config, rng=rng)
            for _ in range(batch_size)
        ]
    return [_sobol_like_candidate(catalog, assets, config=config, stage=config.stage, index=iteration * batch_size + offset) for offset in range(batch_size)]


def _seed_candidates_lite(
    catalog: list[str],
    assets: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
    method: str,
    limit: int,
) -> list[WeeklyMultiAssetCandidate]:
    stage_catalog = [spec for index, spec in enumerate(catalog) if index % config.total_stages == config.stage] or catalog
    universes = _asset_universes(assets)
    candidates: list[WeeklyMultiAssetCandidate] = []
    selector_shift = _stateful_method_offset(method) % len(WEEKLY_ASSET_SELECTORS)
    scales = (0.35, 0.75, 1.15, 1.55)
    for spec_index, spec in enumerate(stage_catalog[: max(1, limit // 2)]):
        universe = universes[spec_index % len(universes)]
        selector = WEEKLY_ASSET_SELECTORS[(spec_index + selector_shift) % len(WEEKLY_ASSET_SELECTORS)]
        candidates.append(WeeklyMultiAssetCandidate((spec,), universe, selector=selector, scale=float(scales[spec_index % len(scales)])))
    while len(candidates) < limit:
        candidates.append(_random_candidate(stage_catalog, assets, config=config, rng=rng))
    return candidates[:limit]


def _beam_candidates(
    catalog: list[str],
    assets: list[str],
    rows: list[dict[str, object]],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
    batch_size: int,
) -> list[WeeklyMultiAssetCandidate]:
    beam = [_candidate_from_row(row) for row in _select_beam(rows, max(1, config.beam_width))]
    if not beam:
        return [_random_candidate(catalog, assets, config=config, rng=rng) for _ in range(batch_size)]
    return [_mutate_candidate(beam[index % len(beam)], catalog, assets, config=config, rng=rng) for index in range(batch_size)]


def _genetic_candidates(
    catalog: list[str],
    assets: list[str],
    rows: list[dict[str, object]],
    prior: list[WeeklyMultiAssetCandidate],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
    batch_size: int,
) -> list[WeeklyMultiAssetCandidate]:
    parents = [_candidate_from_row(row) for row in _select_beam(rows, max(2, config.beam_width))]
    parents.extend(prior[: max(0, config.beam_width - len(parents))])
    if len(parents) < 2:
        return [_random_candidate(catalog, assets, config=config, rng=rng) for _ in range(batch_size)]
    children = []
    for _ in range(batch_size):
        left = parents[int(rng.integers(0, len(parents)))]
        right = parents[int(rng.integers(0, len(parents)))]
        children.append(_mutate_candidate(_crossover(left, right, rng), catalog, assets, config=config, rng=rng))
    return children


def _dehb_candidates(
    catalog: list[str],
    assets: list[str],
    rows: list[dict[str, object]],
    prior: list[WeeklyMultiAssetCandidate],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
    batch_size: int,
) -> list[WeeklyMultiAssetCandidate]:
    elites = [_candidate_from_row(row) for row in _select_beam(rows, max(3, config.beam_width))]
    elites.extend(prior[: max(0, config.beam_width - len(elites))])
    if len(elites) < 3:
        return [_random_candidate(catalog, assets, config=config, rng=rng) for _ in range(batch_size)]
    children = []
    for index in range(batch_size):
        base = elites[int(rng.integers(0, len(elites)))]
        donor = elites[int(rng.integers(0, len(elites)))]
        trial = WeeklyMultiAssetCandidate(
            tuple(sorted(set(base.specs + donor.specs)))[: config.max_features],
            base.assets if index % 2 else donor.assets,
            selector=base.selector if index % 3 else donor.selector,
            intercept=float(np.clip(base.intercept + 0.5 * (donor.intercept - base.intercept), -0.95, 0.95)),
            scale=float(np.clip(base.scale + 0.5 * (donor.scale - base.scale), 0.05, 1.75)),
            smoothing=float(np.clip(base.smoothing + 0.5 * (donor.smoothing - base.smoothing), 0.0, 0.80)),
        )
        children.append(_mutate_candidate(trial, catalog, assets, config=config, rng=rng))
    return children


def _sobol_like_candidate(
    catalog: list[str],
    assets: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    stage: int,
    index: int,
) -> WeeklyMultiAssetCandidate:
    rng = np.random.default_rng(10_000_019 + stage * 1_000_003 + index * 9_176)
    return _random_candidate(catalog, assets, config=config, rng=rng)


def _next_ml_candidates(
    feature_names: list[str],
    universes: list[tuple[str, ...]],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
    stage: int,
    iteration: int,
) -> list[WeeklyMachineLearningCandidate]:
    batch_size = min(max(24, min(256, config.seed_pool // 20 or 24)), max(1, int(config.seed_pool)))
    models = ("ridge", "random_forest", "hist_gradient_boosting")
    candidates: list[WeeklyMachineLearningCandidate] = []
    max_features = max(1, min(config.max_features, len(feature_names)))
    for index in range(batch_size):
        model = models[(stage + iteration + index) % len(models)]
        size = int(rng.integers(1, max_features + 1))
        features = (
            tuple(feature_names)
            if len(feature_names) <= size
            else tuple(sorted(rng.choice(feature_names, size=size, replace=False).tolist()))
        )
        universe = tuple(sorted(set(universes[int(rng.integers(0, len(universes)))])))
        candidates.append(
            WeeklyMachineLearningCandidate(
                features=features,
                assets=universe,
                selector=str(rng.choice(WEEKLY_ASSET_SELECTORS)),
                model=model,
                alpha=float(rng.choice([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])),
                n_estimators=int(rng.choice([32, 48, 64, 96])),
                max_depth=int(rng.choice([2, 3, 4, 5])),
                learning_rate=float(rng.choice([0.03, 0.05, 0.08, 0.12])),
                scale=float(rng.uniform(0.35, 2.0)),
                intercept=float(rng.uniform(-0.35, 0.35)),
                random_seed=int(rng.integers(1, 2_000_000_000)),
            )
        )
    return candidates


def _run_optuna_real(
    catalog: list[str],
    universes: list[tuple[str, ...]],
    *,
    config: MonthlyRiskSearchConfig,
    deadline: float,
    rng: np.random.Generator,
    evaluate: Any,
    sampler_name: str,
    pruner_name: str,
) -> None:
    import optuna

    seed = int(rng.integers(1, 2_000_000_000))
    if sampler_name == "qmc":
        sampler = optuna.samplers.QMCSampler(qmc_type="sobol", seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, constant_liar=True)
    if pruner_name == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=3, reduction_factor=3)
    else:
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial: Any) -> float:
        candidate = _candidate_from_hpo_config(_trial_to_hpo_config(trial, catalog, universes, config), catalog, universes, config)
        score = evaluate(candidate, backend=f"optuna_{sampler_name}_{pruner_name}")
        trial.report(score, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()
        trial.report(score, step=3)
        return score

    timeout = max(1.0, deadline - time.monotonic())
    study.optimize(objective, timeout=timeout, n_trials=max(1, int(config.seed_pool)), catch=(Exception,))


def _run_dehb_real(
    catalog: list[str],
    universes: list[tuple[str, ...]],
    *,
    config: MonthlyRiskSearchConfig,
    deadline: float,
    evaluate: Any,
) -> None:
    from dehb import DEHB

    cs = _hpo_configspace(catalog, universes, config)

    def objective(hpo_config: Any, fidelity: float = 1.0, **_: object) -> dict[str, float]:
        candidate = _candidate_from_hpo_config(hpo_config, catalog, universes, config)
        score = evaluate(candidate, backend="dehb")
        return {"fitness": -score, "cost": float(max(fidelity, 1.0))}

    with tempfile.TemporaryDirectory(prefix="weekly_dehb_") as tmp:
        optimizer = DEHB(
            f=objective,
            cs=cs,
            dimensions=len(cs.get_hyperparameters()),
            min_fidelity=1,
            max_fidelity=3,
            eta=3,
            n_workers=1,
            output_path=tmp,
        )
        optimizer.run(total_cost=max(1.0, deadline - time.monotonic()))


def _run_bohb_real(
    catalog: list[str],
    universes: list[tuple[str, ...]],
    *,
    config: MonthlyRiskSearchConfig,
    deadline: float,
    evaluate: Any,
    run_id: str,
) -> None:
    import hpbandster.core.nameserver as hpns
    from hpbandster.core.worker import Worker
    from hpbandster.optimizers import BOHB

    cs = _hpo_configspace(catalog, universes, config)

    class WeeklyBOHBWorker(Worker):
        def compute(self, config: dict[str, object], budget: float, **_: object) -> dict[str, object]:
            candidate = _candidate_from_hpo_config(config, catalog, universes, config_outer)
            score = evaluate(candidate, backend="hpbandster_bohb")
            return {"loss": float(-score), "info": {"score": float(score), "budget": float(budget)}}

    config_outer = config
    nameserver = hpns.NameServer(run_id=run_id, host="127.0.0.1", port=0)
    ns_host, ns_port = nameserver.start()
    worker = WeeklyBOHBWorker(nameserver=ns_host, nameserver_port=ns_port, run_id=run_id)
    worker.run(background=True)
    optimizer = BOHB(configspace=cs, run_id=run_id, nameserver=ns_host, nameserver_port=ns_port, min_budget=1, max_budget=3)
    try:
        iterations = max(1, min(32, int(config.seed_pool // 32) or 1))
        optimizer.run(n_iterations=iterations, min_n_workers=1)
    finally:
        optimizer.shutdown(shutdown_workers=True)
        nameserver.shutdown()


def _run_smac_real(
    catalog: list[str],
    universes: list[tuple[str, ...]],
    *,
    config: MonthlyRiskSearchConfig,
    deadline: float,
    evaluate: Any,
    seed: int,
) -> None:
    from smac import MultiFidelityFacade, Scenario

    cs = _hpo_configspace(catalog, universes, config)

    def objective(hpo_config: Any, seed: int = 0, budget: float = 1.0) -> float:
        del seed, budget
        candidate = _candidate_from_hpo_config(hpo_config, catalog, universes, config)
        score = evaluate(candidate, backend="smac_multi_fidelity")
        return float(-score)

    with tempfile.TemporaryDirectory(prefix="weekly_smac_") as tmp:
        scenario = Scenario(
            configspace=cs,
            output_directory=Path(tmp),
            deterministic=True,
            n_trials=max(1, int(config.seed_pool)),
            walltime_limit=max(1, int(deadline - time.monotonic())),
            min_budget=1,
            max_budget=3,
            seed=seed,
        )
        smac = MultiFidelityFacade(scenario=scenario, target_function=objective, overwrite=True)
        smac.optimize()


def _trial_to_hpo_config(
    trial: Any,
    catalog: list[str],
    universes: list[tuple[str, ...]],
    config: MonthlyRiskSearchConfig,
) -> dict[str, object]:
    data: dict[str, object] = {
        "feature_count": trial.suggest_int("feature_count", 1, max(1, min(config.max_features, len(catalog)))),
        "asset_universe": trial.suggest_int("asset_universe", 0, max(0, len(universes) - 1)),
        "selector": trial.suggest_categorical("selector", list(WEEKLY_ASSET_SELECTORS)),
        "intercept": trial.suggest_float("intercept", -0.55, 0.55),
        "scale": trial.suggest_float("scale", 0.15, 1.45),
        "smoothing": trial.suggest_categorical("smoothing", [0.0, 0.10, 0.20, 0.35, 0.50]),
    }
    max_index = max(0, len(catalog) - 1)
    for index in range(max(1, config.max_features)):
        data[f"spec_{index}"] = trial.suggest_int(f"spec_{index}", 0, max_index)
    return data


def _hpo_configspace(
    catalog: list[str],
    universes: list[tuple[str, ...]],
    config: MonthlyRiskSearchConfig,
) -> Any:
    from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter

    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter("feature_count", lower=1, upper=max(1, min(config.max_features, len(catalog)))))
    cs.add_hyperparameter(UniformIntegerHyperparameter("asset_universe", lower=0, upper=max(0, len(universes) - 1)))
    cs.add_hyperparameter(CategoricalHyperparameter("selector", choices=list(WEEKLY_ASSET_SELECTORS)))
    cs.add_hyperparameter(UniformFloatHyperparameter("intercept", lower=-0.55, upper=0.55))
    cs.add_hyperparameter(UniformFloatHyperparameter("scale", lower=0.15, upper=1.45))
    cs.add_hyperparameter(CategoricalHyperparameter("smoothing", choices=[0.0, 0.10, 0.20, 0.35, 0.50]))
    max_index = max(0, len(catalog) - 1)
    for index in range(max(1, config.max_features)):
        cs.add_hyperparameter(UniformIntegerHyperparameter(f"spec_{index}", lower=0, upper=max_index))
    return cs


def _candidate_from_hpo_config(
    hpo_config: Any,
    catalog: list[str],
    universes: list[tuple[str, ...]],
    config: MonthlyRiskSearchConfig,
) -> WeeklyMultiAssetCandidate:
    values = dict(hpo_config)
    feature_count = int(values.get("feature_count", 1) or 1)
    specs = []
    for index in range(max(1, min(config.max_features, feature_count))):
        spec_index = int(float(values.get(f"spec_{index}", 0) or 0)) % len(catalog)
        specs.append(catalog[spec_index])
    universe_index = int(float(values.get("asset_universe", 0) or 0)) % len(universes)
    smoothing = float(values.get("smoothing", 0.0) or 0.0)
    return WeeklyMultiAssetCandidate(
        tuple(sorted(set(specs))) or (catalog[0],),
        tuple(sorted(set(universes[universe_index]))),
        selector=str(values.get("selector", "momentum_26w") or "momentum_26w"),
        intercept=float(values.get("intercept", 0.0) or 0.0),
        scale=float(values.get("scale", 1.0) or 1.0),
        smoothing=smoothing,
    )


def _evaluate_and_stamp(
    examples: pd.DataFrame,
    candidates: list[WeeklyMultiAssetCandidate],
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str, float, float, float]],
    *,
    method: str,
    wave: int,
    stage: int,
    start: float,
    score_mode: str = "train_only_weekly_sp500_down_5pct",
) -> list[dict[str, object]]:
    rows = _evaluate_unique(examples, candidates, seen, method=method, score_mode=score_mode)
    stamped = []
    for index, row in enumerate(rows, start=1):
        elapsed = max(0.0, time.monotonic() - start)
        row["method"] = method
        row["wave"] = int(wave)
        row["stage"] = int(stage)
        row["elapsed_seconds"] = float(elapsed)
        row["candidates_evaluated"] = int(index)
        row["first_seen_wave"] = int(wave)
        row["first_seen_minute"] = float(elapsed / 60.0)
        row["accepted"] = bool(row.get("accepted")) and _has_expected_train_counts(row)
        if score_mode == "train_calmar_max_validation_80pct_report":
            row["verified_calmar_similarity"] = (
                bool(row.get("accepted"))
                and _has_expected_validation_counts(row)
                and bool(row.get("verified_calmar_similarity"))
            )
            row["verified_train_validation_5pct"] = False
        elif score_mode == WEEKLY_MAX_SHARPE_SCORE_MODE:
            row["verified_sharpe_robust"] = (
                bool(row.get("accepted"))
                and _has_expected_validation_counts(row)
                and bool(row.get("verified_sharpe_robust"))
            )
            row["verified_train_validation_5pct"] = False
        else:
            row["verified_train_validation_5pct"] = bool(row.get("accepted")) and _has_expected_validation_counts(row) and bool(row.get("verified_train_validation_5pct"))
        row["locked_opened"] = False
        row["validation_role"] = "report_only"
        stamped.append(row)
    return stamped


def _trim_stage_rows(rows: list[dict[str, object]], top_rows: int) -> list[dict[str, object]]:
    sorted_rows = sorted(rows, key=lambda row: float(row.get("weekly_multi_asset_score", 0.0) or 0.0), reverse=True)
    if top_rows <= 0:
        return sorted_rows
    verified = [
        row
        for row in sorted_rows
        if bool(row.get("verified_train_validation_5pct"))
        or bool(row.get("verified_calmar_similarity"))
        or bool(row.get("verified_sharpe_robust"))
    ]
    top = sorted_rows[:top_rows]
    by_id = {str(row["candidate_id"]): row for row in [*top, *verified]}
    return sorted(by_id.values(), key=lambda row: float(row.get("weekly_multi_asset_score", 0.0) or 0.0), reverse=True)


def _strict_verified(rows: pd.DataFrame) -> pd.DataFrame:
    base = _verified(rows)
    if base.empty:
        return base
    if "verified_sharpe_robust" in base and base["verified_sharpe_robust"].astype(bool).any():
        mask = (
            base["verified_sharpe_robust"].astype(bool)
            & (pd.to_numeric(base["train_years_total"], errors="coerce") == EXPECTED_TRAIN_YEARS)
            & (pd.to_numeric(base["validation_years_total"], errors="coerce") == EXPECTED_VALIDATION_YEARS)
            & ~base["locked_opened"].astype(bool)
        )
        return base.loc[mask].copy()
    mask = (
        (pd.to_numeric(base["train_years_total"], errors="coerce") == EXPECTED_TRAIN_YEARS)
        & (pd.to_numeric(base["validation_years_total"], errors="coerce") == EXPECTED_VALIDATION_YEARS)
        & (pd.to_numeric(base["train_down_years_total"], errors="coerce") == EXPECTED_TRAIN_DOWN_YEARS)
        & (pd.to_numeric(base["validation_down_years_total"], errors="coerce") == EXPECTED_VALIDATION_DOWN_YEARS)
    )
    return base.loc[mask].copy()


def _has_expected_train_counts(row: dict[str, object]) -> bool:
    return (
        int(row.get("train_years_total", 0) or 0) == EXPECTED_TRAIN_YEARS
        and int(row.get("train_down_years_total", 0) or 0) == EXPECTED_TRAIN_DOWN_YEARS
    )


def _has_expected_validation_counts(row: dict[str, object]) -> bool:
    return (
        int(row.get("validation_years_total", 0) or 0) == EXPECTED_VALIDATION_YEARS
        and int(row.get("validation_down_years_total", 0) or 0) == EXPECTED_VALIDATION_DOWN_YEARS
    )


def _state_candidates(state: dict[str, object]) -> list[WeeklyMultiAssetCandidate]:
    candidates = []
    for row in state.get("candidates", []) if isinstance(state, dict) else []:
        try:
            candidates.append(
                WeeklyMultiAssetCandidate(
                    tuple(row.get("specs", [])),
                    tuple(row.get("assets", ["SPY"])),
                    selector=str(row.get("selector", "momentum_26w")),
                    intercept=float(row.get("intercept", 0.0)),
                    scale=float(row.get("scale", 1.0)),
                    smoothing=float(row.get("smoothing", 0.0)),
                )
            )
        except (TypeError, ValueError):
            continue
    return candidates


def _state_candidate_from_row(row: dict[str, object]) -> dict[str, object]:
    candidate = _candidate_from_row(row)
    return {
        "candidate_id": _candidate_id(candidate),
        "specs": list(candidate.specs),
        "assets": list(candidate.assets),
        "selector": candidate.selector,
        "intercept": float(candidate.intercept),
        "scale": float(candidate.scale),
        "smoothing": float(candidate.smoothing),
        "train_score": float(row.get("weekly_multi_asset_score", 0.0) or 0.0),
        "train_years_positive": int(row.get("train_years_positive", 0) or 0),
        "train_years_total": int(row.get("train_years_total", 0) or 0),
        "train_down_years_ge_5pct": int(row.get("train_down_years_ge_5pct", 0) or 0),
        "train_down_years_total": int(row.get("train_down_years_total", 0) or 0),
    }


def _dedupe_state_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    dedup: dict[str, dict[str, object]] = {}
    for row in candidates:
        key = str(row.get("candidate_id") or json.dumps(row, sort_keys=True))
        current = dedup.get(key)
        if current is None or float(row.get("train_score", 0.0) or 0.0) > float(current.get("train_score", 0.0) or 0.0):
            dedup[key] = row
    return list(dedup.values())


def _sanitize_state_candidate(row: dict[str, object]) -> dict[str, object]:
    return {
        "candidate_id": str(row.get("candidate_id", "")),
        "specs": list(row.get("specs", [])),
        "assets": list(row.get("assets", ["SPY"])),
        "selector": str(row.get("selector", "momentum_26w")),
        "intercept": float(row.get("intercept", 0.0) or 0.0),
        "scale": float(row.get("scale", 1.0) or 1.0),
        "smoothing": float(row.get("smoothing", 0.0) or 0.0),
        "train_score": float(row.get("train_score", 0.0) or 0.0),
        "train_years_positive": int(row.get("train_years_positive", 0) or 0),
        "train_years_total": int(row.get("train_years_total", 0) or 0),
        "train_down_years_ge_5pct": int(row.get("train_down_years_ge_5pct", 0) or 0),
        "train_down_years_total": int(row.get("train_down_years_total", 0) or 0),
    }


def _stateful_method_summary(rows: pd.DataFrame, verified: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "method" not in rows:
        return pd.DataFrame()
    records = []
    for method, group in rows.groupby("method", sort=True):
        group = group.sort_values("weekly_multi_asset_score", ascending=False)
        method_verified = verified.loc[verified["method"] == method] if "method" in verified else pd.DataFrame()
        records.append(
            {
                "method": method,
                "rows": int(len(group)),
                "verified_count": int(len(method_verified)),
                "unique_verified_count": int(method_verified["candidate_id"].nunique()) if "candidate_id" in method_verified else 0,
                "best_candidate": str(group.iloc[0].get("candidate_id", "")) if not group.empty else None,
                "best_score_720m": float(group.iloc[0].get("weekly_multi_asset_score", np.nan)) if not group.empty else None,
                "best_validation_min_year_return": _best_float(method_verified, "validation_min_year_return"),
                "best_validation_down_min_return": _best_float(method_verified, "validation_down_min_return"),
                "best_validation_cagr": _best_float(method_verified, "validation_cagr"),
                "best_validation_mdd": _best_float(method_verified, "validation_mdd", highest=False),
                "best_train_calmar": _best_float(group, "train_calmar"),
                "best_validation_calmar": _best_float(method_verified, "validation_calmar"),
                "best_validation_calmar_ratio_to_train": _best_float(method_verified, "validation_calmar_ratio_to_train"),
                "best_train_sharpe": _best_float(group, "train_sharpe"),
                "best_validation_sharpe": _best_float(method_verified, "validation_sharpe"),
                "best_validation_sharpe_ratio_to_train": _best_float(method_verified, "validation_sharpe_ratio_to_train"),
            }
        )
    return pd.DataFrame(records)


def _stateful_efficiency(rows: pd.DataFrame, verified: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "method" not in rows:
        return pd.DataFrame()
    records = []
    for method, group in rows.groupby("method", sort=True):
        method_verified = verified.loc[verified["method"] == method] if "method" in verified else pd.DataFrame()
        max_elapsed = float(pd.to_numeric(group.get("elapsed_seconds", pd.Series(dtype=float)), errors="coerce").max() or 0.0)
        hours = max(max_elapsed / 3600.0, 1e-9)
        first = None
        if not method_verified.empty and "first_seen_minute" in method_verified:
            first = float(pd.to_numeric(method_verified["first_seen_minute"], errors="coerce").min())
        records.append(
            {
                "method": method,
                "rows": int(len(group)),
                "verified_count": int(len(method_verified)),
                "verified_per_hour": float(len(method_verified) / hours),
                "time_to_first_verified": first,
                "best_score_240m": _best_score_before(group, 240.0),
                "best_score_480m": _best_score_before(group, 480.0),
                "best_score_720m": _best_score_before(group, 720.0),
            }
        )
    return pd.DataFrame(records)


def _best_score_before(group: pd.DataFrame, minute: float) -> float | None:
    if "first_seen_minute" not in group:
        return None
    subset = group.loc[pd.to_numeric(group["first_seen_minute"], errors="coerce") <= minute]
    if subset.empty:
        return None
    return float(pd.to_numeric(subset["weekly_multi_asset_score"], errors="coerce").max())


def _best_float(rows: pd.DataFrame, column: str, *, highest: bool = True) -> float | None:
    if rows.empty or column not in rows:
        return None
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max() if highest else values.min())


def _unique_stage_count(rows: pd.DataFrame) -> int:
    if rows.empty or "method" not in rows or "stage" not in rows:
        return 0
    if "wave" in rows:
        return int(rows[["method", "wave", "stage"]].drop_duplicates().shape[0])
    return int(rows[["method", "stage"]].drop_duplicates().shape[0])


def _failed_stage_count(rows: pd.DataFrame) -> int:
    if rows.empty or "stage_failed" not in rows:
        return 0
    failed = rows.loc[rows["stage_failed"].astype(str).str.lower().isin(("true", "1"))]
    return _unique_stage_count(failed)


def _stage_summary(rows: list[dict[str, object]], *, method: str, wave: int, stage: int) -> dict[str, object]:
    return {
        "method": method,
        "wave": int(wave),
        "stage": int(stage),
        "rows": int(len(rows)),
        "accepted": int(sum(bool(row.get("accepted")) for row in rows)),
        "verified_train_validation_5pct": int(sum(bool(row.get("verified_train_validation_5pct")) for row in rows)),
        "verified_calmar_similarity": int(sum(bool(row.get("verified_calmar_similarity")) for row in rows)),
        "verified_sharpe_robust": int(sum(bool(row.get("verified_sharpe_robust")) for row in rows)),
        "best_candidate": str(rows[0].get("candidate_id")) if rows else None,
        "locked_opened": False,
        "score_mode": str(rows[0].get("score_mode", "")) if rows else "",
        "validation_role": "report_only",
    }


def _empty_state(method: str, wave: int, stage: int) -> dict[str, object]:
    return {
        "method": method,
        "wave": int(wave),
        "stage": int(stage),
        "candidate_count": 0,
        "candidates": [],
        "validation_role": "report_only",
        "locked_opened": False,
    }


def _stateful_method_offset(method: str) -> int:
    return {
        "sobol_random_asha": 101_000,
        "tpe_asha_lite": 113_000,
        "dehb_lite": 127_000,
        "bohb_lite": 131_000,
        "smac_mf_lite": 137_000,
        "sobol_random_asha_real": 141_000,
        "optuna_tpe_hyperband": 143_000,
        "dehb_real": 145_000,
        "bohb_real": 147_000,
        "smac_mf_real": 148_000,
        "beam": 149_000,
        "genetic": 157_000,
        "bandit": 161_000,
        "machine_learning": 163_000,
    }.get(method, 0)


def _engine_method(method: str) -> str:
    return METHOD_ENGINE_ALIASES.get(method, method)
