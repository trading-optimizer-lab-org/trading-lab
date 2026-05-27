"""Train-first ML strategy search for Aurora.

The search trains only on the train window, examines validation only after a
candidate passes the train target, and never loads locked data.
"""
from __future__ import annotations

import json
import importlib
import math
import random
import time
import traceback
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurora.core.metrics import compute_metrics
from aurora.core.runtime_paths import base_data_dir
from aurora.data_contracts.timeseries_store import TimeSeriesStore
from aurora.ml.features_pipeline import FeaturePipeline, FeaturePipelineConfig
from aurora.validation.deflated_sharpe import deflated_sharpe_annualized


ALLOWED_SOURCE_COLUMNS = {"open", "high", "low", "close", "adj_close", "volume"}
REQUIRED_SOURCE_COLUMNS = {"open", "high", "low", "close", "adj_close"}
OHLC_COLUMNS = ("open", "high", "low", "close")
FORBIDDEN_LOCKED_COLUMNS = ("locked", "oos_locked", "future_return")

_X_TRAIN: np.ndarray | None = None
_Y_TRAIN: np.ndarray | None = None
_RET_TRAIN_NEXT: np.ndarray | None = None
_RET_VALID_NEXT: np.ndarray | None = None
_X_VALID: np.ndarray | None = None
_TRAIN_SUBPERIOD_MASKS: tuple[np.ndarray, ...] = tuple()
_TRAIN_ANNUAL_RETURN_MASKS: tuple[np.ndarray, ...] = tuple()
_VALID_SUBPERIOD_MASKS: tuple[np.ndarray, ...] = tuple()
_VALID_ANNUAL_RETURN_MASKS: tuple[np.ndarray, ...] = tuple()
_TRAIN_YEARS: float = 0.0
_VALID_YEARS: float = 0.0


@dataclass(frozen=True)
class MLSearchConfig:
    run_id: str
    symbol: str = "SPY"
    library: str = "prices_daily"
    target_calmar: float = 1.0
    validation_target_calmar: float | None = 1.0
    train_end: str = "2013-10-18"
    validation_start: str = "2013-10-21"
    validation_end: str = "2020-01-28"
    locked_start: str = "2020-01-29"
    workers: int = 6
    max_candidates: int = 5000
    batch_size: int = 600
    seed: int = 42
    run_root: str | None = None
    no_costs: bool = True
    no_locked: bool = True
    include_kronos: bool = False
    include_classic_ml: bool = True
    include_sequence_models: bool = False
    include_pending_features: bool = False
    pending_feature_library: str = "features_pending_daily"
    pending_feature_version: str = "pending_features_v1"
    models: tuple[str, ...] = ("lightgbm", "xgboost")
    top_n: int = 25
    target_objective_count: int = 1
    min_feature_jaccard_distance: float = 0.15
    min_behavior_distance: float = 0.15
    train_subperiod_count: int = 4
    validation_subperiod_count: int | None = None
    min_train_subperiod_calmar: float = 0.0
    min_validation_subperiod_calmar: float | None = None
    min_train_cagr: float | None = None
    min_validation_cagr: float | None = None
    max_train_mdd: float | None = None
    max_validation_mdd: float | None = None
    max_train_calmar: float | None = None
    min_train_annual_return: float | None = None
    min_validation_annual_return: float | None = None
    min_train_annual_calmar: float | None = None
    min_validation_annual_calmar: float | None = None
    max_train_validation_calmar_ratio: float | None = None
    min_validation_excess_pvalue: float | None = None
    min_validation_bootstrap_calmar_p05: float | None = None
    min_validation_bootstrap_excess_calmar_p05: float | None = None
    max_validation_random_baseline_pvalue: float | None = None
    min_validation_deflated_sharpe: float | None = None
    max_validation_pbo: float | None = None
    min_feature_ablation_validation_calmar: float | None = None
    min_validation_regime_calmar: float | None = None
    max_validation_trade_concentration: float | None = None
    statistical_bootstrap_paths: int = 300
    statistical_bootstrap_block: int = 21
    statistical_random_shuffles: int = 300
    statistical_pbo_splits: int = 8
    min_trades_per_year: float = 0.5
    max_trades_per_year: float | None = None
    min_long_fraction: float | None = None
    max_long_fraction: float | None = None
    always_long_threshold: float = 0.98
    always_short_threshold: float = 0.02
    max_features_per_candidate: int | None = None
    complexity_penalty: float = 0.0
    anti_overfit_ledger_path: str | None = None
    family_tournament_mode: bool = False
    early_stability_screen: bool = False
    simple_survivors_mode: bool = False
    reject_same_feature_family: bool = False
    adaptive_family_search: bool = False
    adaptive_quick_screen_candidates: int = 0
    adaptive_family_min_weight: float = 0.25
    adaptive_family_reward: float = 4.0
    adaptive_initial_family_attempts: dict[str, int] | None = None
    adaptive_initial_family_rewards: dict[str, float] | None = None
    penalized_feature_pools: tuple[str, ...] = tuple()
    penalized_feature_pool_factor: float = 0.25
    defer_robustness_until_basic_pass: bool = False
    effective_dsr_trials: int | None = None
    time_limit_seconds: float | None = None
    literature_ideas: tuple[dict[str, Any], ...] = tuple()


@dataclass(frozen=True)
class MLSearchMetrics:
    calmar: float
    cagr: float
    mdd: float
    trades: int
    trades_per_year: float
    long_fraction: float
    final_nav: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MLSearchCandidate:
    candidate_id: str
    route: str
    model: str
    feature_set: tuple[str, ...]
    threshold: float
    direction: int
    train_metrics: MLSearchMetrics
    validation_metrics: MLSearchMetrics | None
    rule: str
    train_subperiod_metrics: tuple[MLSearchMetrics, ...] = tuple()
    validation_subperiod_metrics: tuple[MLSearchMetrics, ...] = tuple()
    positions_signature: str = ""
    behavior_vector: tuple[float, float, float] = tuple()
    horizon: int = 1
    target_type: str = "direction"
    smoothing: int = 1
    model_params: dict[str, Any] | None = None
    feature_importance: dict[str, float] | None = None
    seed: int | None = None
    robustness: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "route": self.route,
            "model": self.model,
            "feature_set": list(self.feature_set),
            "threshold": self.threshold,
            "direction": self.direction,
            "train_metrics": self.train_metrics.to_dict(),
            "validation_metrics": None
            if self.validation_metrics is None
            else self.validation_metrics.to_dict(),
            "rule": self.rule,
            "positions_signature": self.positions_signature,
            "behavior_vector": list(self.behavior_vector),
            "horizon": self.horizon,
            "target_type": self.target_type,
            "smoothing": self.smoothing,
            "model_params": {} if self.model_params is None else dict(self.model_params),
            "seed": self.seed,
            "robustness": {} if self.robustness is None else dict(self.robustness),
            "feature_importance": {}
            if self.feature_importance is None
            else dict(self.feature_importance),
            "train_subperiod_metrics": [
                metrics.to_dict() for metrics in self.train_subperiod_metrics
            ],
            "validation_subperiod_metrics": [
                metrics.to_dict() for metrics in self.validation_subperiod_metrics
            ],
        }


@dataclass(frozen=True)
class MLSearchReport:
    status: str
    locked_opened: bool
    objective_met: bool
    run_id: str
    output_dir: str
    symbol: str
    workers: int
    candidates_evaluated: int
    batches_completed: int
    train_period: tuple[str, str]
    validation_period: tuple[str, str]
    locked_period: tuple[str, str]
    used_columns: tuple[str, ...]
    route_errors: tuple[str, ...]
    best_train: MLSearchCandidate | None
    best_validation: MLSearchCandidate | None
    objective_candidates: tuple[MLSearchCandidate, ...]
    top: tuple[MLSearchCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "locked_opened": self.locked_opened,
            "objective_met": self.objective_met,
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "symbol": self.symbol,
            "workers": self.workers,
            "candidates_evaluated": self.candidates_evaluated,
            "batches_completed": self.batches_completed,
            "train_period": self.train_period,
            "validation_period": self.validation_period,
            "locked_period": self.locked_period,
            "used_columns": list(self.used_columns),
            "route_errors": list(self.route_errors),
            "best_train": None if self.best_train is None else self.best_train.to_dict(),
            "best_validation": None
            if self.best_validation is None
            else self.best_validation.to_dict(),
            "objective_candidates": [
                candidate.to_dict() for candidate in self.objective_candidates
            ],
            "objective_candidates_found": len(self.objective_candidates),
            "top": [candidate.to_dict() for candidate in self.top],
        }


def run_ml_search(config: MLSearchConfig) -> MLSearchReport:
    if not config.no_costs:
        raise ValueError("ml-search v1 only supports --no-costs")
    if not config.no_locked:
        raise ValueError("ml-search v1 requires --no-locked")
    if config.workers < 1:
        raise ValueError("workers must be >= 1")
    if config.max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if config.target_objective_count < 1:
        raise ValueError("target_objective_count must be >= 1")
    if config.train_subperiod_count < 1:
        raise ValueError("train_subperiod_count must be >= 1")
    validation_subperiod_count = (
        config.train_subperiod_count
        if config.validation_subperiod_count is None
        else int(config.validation_subperiod_count)
    )
    if validation_subperiod_count < 1:
        raise ValueError("validation_subperiod_count must be >= 1")
    models = _parse_models(config.models)

    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    progress_path = output_dir / "progress.jsonl"
    candidates_path = output_dir / "candidates.jsonl"
    best_path = output_dir / "best_candidates.json"
    best_md_path = output_dir / "best_candidates.md"
    stderr_path = output_dir / "stderr.log"
    feature_sets_path = output_dir / "feature_sets.json"
    validation_exams_path = output_dir / "validation_exams.jsonl"
    rejected_path = output_dir / "rejected_candidates.jsonl"
    model_importance_path = output_dir / "model_importance.jsonl"
    objective_artifacts_dir = output_dir / "objective_artifacts"
    for path in (
        status_path,
        progress_path,
        candidates_path,
        best_path,
        best_md_path,
        stderr_path,
        feature_sets_path,
        validation_exams_path,
        rejected_path,
        model_importance_path,
    ):
        if path.exists():
            path.unlink()
    stderr_path.write_text("", encoding="utf-8")
    objective_artifacts_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    route_errors: list[str] = []
    candidates_evaluated = 0
    batches_completed = 0
    top: list[MLSearchCandidate] = []
    best_validation: MLSearchCandidate | None = None
    objective_candidates: list[MLSearchCandidate] = []
    objective_signatures: set[str] = set()
    objective_met = False
    time_limit_reached = False
    rejection_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()

    _write_json(status_path, _status_payload(config, output_dir, "running"))

    try:
        frame = load_ml_frame(config.symbol, library=config.library, end=config.validation_end)
        train, validation = _split_frame(frame, config)
        features = build_feature_frame(frame)
        if config.include_pending_features:
            pending_features = load_pending_feature_frame(
                config.symbol,
                library=config.pending_feature_library,
                version=config.pending_feature_version,
                end=config.validation_end,
            )
            features = join_pending_features(features, pending_features)
        train_features = features.loc[train.index]
        validation_features = features.loc[validation.index]
        train_next_returns = _next_returns(train["close"])
        validation_next_returns = _next_returns(validation["close"])
        train_xy = _aligned_xy(train_features, train_next_returns)
        validation_xy = _aligned_validation(validation_features, validation_next_returns)
        if len(train_xy[0]) < 50:
            raise ValueError("not enough train rows after feature alignment")
        train_subperiod_masks = _subperiod_masks(train_xy[4], config.train_subperiod_count)
        train_annual_return_masks = _annual_return_masks(train_xy[4])
        validation_index = _aligned_index(validation_features, validation_next_returns)
        validation_subperiod_masks = _subperiod_masks(validation_index, validation_subperiod_count)
        validation_annual_return_masks = _annual_return_masks(validation_index)

        feature_groups = _feature_groups(list(train_xy[2]))
        literature_groups = _literature_feature_groups(
            list(train_xy[2]),
            config.literature_ideas,
        )
        feature_groups.update(literature_groups)
        _write_json(feature_sets_path, feature_groups)
        anti_overfit_patterns = _load_anti_overfit_patterns(config.anti_overfit_ledger_path)
        group_names = [
            name for name in feature_groups if name != "all" and feature_groups[name]
        ]
        specs: list[dict[str, Any]] = []
        if not config.adaptive_family_search:
            specs = _candidate_specs(
                random.Random(config.seed),
                feature_groups,
                max(1, int(config.max_candidates)),
                include_classic_ml=config.include_classic_ml,
                models=models,
                target_calmar=config.target_calmar,
                validation_target_calmar=config.validation_target_calmar,
                min_train_subperiod_calmar=config.min_train_subperiod_calmar,
                min_validation_subperiod_calmar=config.min_validation_subperiod_calmar,
                min_train_cagr=config.min_train_cagr,
                min_validation_cagr=config.min_validation_cagr,
                max_train_mdd=config.max_train_mdd,
                max_validation_mdd=config.max_validation_mdd,
                max_train_calmar=config.max_train_calmar,
                min_train_annual_return=config.min_train_annual_return,
                min_validation_annual_return=config.min_validation_annual_return,
                min_train_annual_calmar=config.min_train_annual_calmar,
                min_validation_annual_calmar=config.min_validation_annual_calmar,
                max_train_validation_calmar_ratio=config.max_train_validation_calmar_ratio,
                min_validation_excess_pvalue=config.min_validation_excess_pvalue,
                min_validation_bootstrap_calmar_p05=config.min_validation_bootstrap_calmar_p05,
                min_validation_bootstrap_excess_calmar_p05=config.min_validation_bootstrap_excess_calmar_p05,
                max_validation_random_baseline_pvalue=config.max_validation_random_baseline_pvalue,
                min_validation_deflated_sharpe=config.min_validation_deflated_sharpe,
                max_validation_pbo=config.max_validation_pbo,
                min_feature_ablation_validation_calmar=config.min_feature_ablation_validation_calmar,
                min_validation_regime_calmar=config.min_validation_regime_calmar,
                max_validation_trade_concentration=config.max_validation_trade_concentration,
                statistical_bootstrap_paths=config.statistical_bootstrap_paths,
                statistical_bootstrap_block=config.statistical_bootstrap_block,
                statistical_random_shuffles=config.statistical_random_shuffles,
                statistical_pbo_splits=config.statistical_pbo_splits,
                min_trades_per_year=config.min_trades_per_year,
                max_trades_per_year=config.max_trades_per_year,
                min_long_fraction=config.min_long_fraction,
                max_long_fraction=config.max_long_fraction,
                always_long_threshold=config.always_long_threshold,
                always_short_threshold=config.always_short_threshold,
                max_features_per_candidate=config.max_features_per_candidate,
                complexity_penalty=config.complexity_penalty,
                anti_overfit_patterns=anti_overfit_patterns,
                early_stability_screen=config.early_stability_screen,
                simple_survivors_mode=config.simple_survivors_mode,
                defer_robustness_until_basic_pass=config.defer_robustness_until_basic_pass,
                effective_dsr_trials=config.effective_dsr_trials,
            )
        batch_size = max(1, int(config.batch_size))
        adaptive_family_attempts: Counter[str] = Counter(
            {
                str(name): int(value)
                for name, value in (config.adaptive_initial_family_attempts or {}).items()
            }
        )
        adaptive_family_rewards: Counter[str] = Counter(
            {
                str(name): float(value)
                for name, value in (config.adaptive_initial_family_rewards or {}).items()
            }
        )
        with ProcessPoolExecutor(
            max_workers=config.workers,
            initializer=_init_worker,
            initargs=(
                train_xy[0],
                train_xy[1],
                train_xy[3],
                validation_xy[0],
                validation_xy[1],
                train_subperiod_masks,
                train_annual_return_masks,
                validation_subperiod_masks,
                validation_annual_return_masks,
                _years_from_rows(len(train_xy[3])),
                _years_from_rows(len(validation_xy[1])),
            ),
        ) as pool:
            batch_start = 0
            while True:
                if config.adaptive_family_search:
                    remaining = int(config.max_candidates) - candidates_evaluated
                    if remaining <= 0:
                        break
                    quick_screen = (
                        int(config.adaptive_quick_screen_candidates) > 0
                        and candidates_evaluated < int(config.adaptive_quick_screen_candidates)
                    )
                    batch_models = (
                        ("corr", "ridge", "logistic")
                        if quick_screen
                        else models
                    )
                    batch = _candidate_specs(
                        random.Random(config.seed + candidates_evaluated + 1),
                        feature_groups,
                        min(batch_size, remaining),
                        include_classic_ml=True,
                        models=batch_models,
                        target_calmar=config.target_calmar,
                        validation_target_calmar=config.validation_target_calmar,
                        min_train_subperiod_calmar=config.min_train_subperiod_calmar,
                        min_validation_subperiod_calmar=config.min_validation_subperiod_calmar,
                        min_train_cagr=config.min_train_cagr,
                        min_validation_cagr=config.min_validation_cagr,
                        max_train_mdd=config.max_train_mdd,
                        max_validation_mdd=config.max_validation_mdd,
                        max_train_calmar=config.max_train_calmar,
                        min_train_annual_return=config.min_train_annual_return,
                        min_validation_annual_return=config.min_validation_annual_return,
                        min_train_annual_calmar=config.min_train_annual_calmar,
                        min_validation_annual_calmar=config.min_validation_annual_calmar,
                        max_train_validation_calmar_ratio=config.max_train_validation_calmar_ratio,
                        min_validation_excess_pvalue=config.min_validation_excess_pvalue,
                        min_validation_bootstrap_calmar_p05=config.min_validation_bootstrap_calmar_p05,
                        min_validation_bootstrap_excess_calmar_p05=config.min_validation_bootstrap_excess_calmar_p05,
                        max_validation_random_baseline_pvalue=config.max_validation_random_baseline_pvalue,
                        min_validation_deflated_sharpe=config.min_validation_deflated_sharpe,
                        max_validation_pbo=config.max_validation_pbo,
                        min_feature_ablation_validation_calmar=config.min_feature_ablation_validation_calmar,
                        min_validation_regime_calmar=config.min_validation_regime_calmar,
                        max_validation_trade_concentration=config.max_validation_trade_concentration,
                        statistical_bootstrap_paths=config.statistical_bootstrap_paths,
                        statistical_bootstrap_block=config.statistical_bootstrap_block,
                        statistical_random_shuffles=config.statistical_random_shuffles,
                        statistical_pbo_splits=config.statistical_pbo_splits,
                        min_trades_per_year=config.min_trades_per_year,
                        max_trades_per_year=config.max_trades_per_year,
                        min_long_fraction=config.min_long_fraction,
                        max_long_fraction=config.max_long_fraction,
                        always_long_threshold=config.always_long_threshold,
                        always_short_threshold=config.always_short_threshold,
                        max_features_per_candidate=config.max_features_per_candidate,
                        complexity_penalty=config.complexity_penalty,
                        anti_overfit_patterns=anti_overfit_patterns,
                        early_stability_screen=config.early_stability_screen,
                        simple_survivors_mode=config.simple_survivors_mode,
                        family_weights=_adaptive_family_weights(
                            group_names,
                            adaptive_family_attempts,
                            adaptive_family_rewards,
                            min_weight=config.adaptive_family_min_weight,
                            reward_scale=config.adaptive_family_reward,
                            penalties={
                                name: config.penalized_feature_pool_factor
                                for name in config.penalized_feature_pools
                            },
                        ),
                        candidate_start=candidates_evaluated,
                        defer_robustness_until_basic_pass=config.defer_robustness_until_basic_pass,
                        effective_dsr_trials=config.effective_dsr_trials,
                    )
                else:
                    if batch_start >= len(specs):
                        break
                    batch = specs[batch_start: batch_start + batch_size]
                    batch_start += batch_size
                rows = list(pool.map(_evaluate_spec, batch))
                batches_completed += 1
                for row in rows:
                    pool_name = str(row.get("spec", {}).get("feature_pool", "all"))
                    if config.adaptive_family_search:
                        adaptive_family_attempts[pool_name] += 1
                        adaptive_family_rewards[pool_name] += _adaptive_family_reward(row)
                    reason = row.get("rejection_reason")
                    if reason:
                        rejection_counts[str(reason)] += 1
                        _append_jsonl(rejected_path, _rejected_payload(row, train_xy[2]))
                        if config.anti_overfit_ledger_path:
                            _append_jsonl(
                                Path(config.anti_overfit_ledger_path),
                                _anti_overfit_ledger_payload(row, train_xy[2]),
                            )
                    bucket = row.get("behavior_bucket")
                    if bucket:
                        behavior_counts[str(bucket)] += 1
                    if row.get("feature_importance"):
                        _append_jsonl(
                            model_importance_path,
                            _importance_payload(row, train_xy[2]),
                        )
                batch_candidates = [
                    _candidate_from_row(row, train_xy[2], config.target_calmar)
                    for row in rows
                    if row.get("candidate") is not None
                ]
                rows_by_id = {str(row["spec"]["candidate_id"]): row for row in rows}
                candidates_evaluated += len(rows)
                for candidate in batch_candidates:
                    _append_jsonl(candidates_path, candidate.to_dict())
                    if candidate.validation_metrics is not None:
                        _append_jsonl(validation_exams_path, candidate.to_dict())
                top = sorted(
                    [*top, *batch_candidates],
                    key=lambda item: item.train_metrics.calmar,
                    reverse=True,
                )[: config.top_n]
                validation_examined = [
                    candidate
                    for candidate in [*top, *batch_candidates]
                    if candidate.validation_metrics is not None
                ]
                if validation_examined:
                    best_validation = sorted(
                        validation_examined,
                        key=lambda item: item.validation_metrics.calmar
                        if item.validation_metrics is not None
                        else -math.inf,
                        reverse=True,
                    )[0]
                validation_passed = [
                    candidate
                    for candidate in batch_candidates
                    if (
                        candidate.validation_metrics is not None
                        and rows_by_id[candidate.candidate_id].get("rejection_reason") is None
                        and (
                            config.validation_target_calmar is None
                            or candidate.validation_metrics.calmar >= config.validation_target_calmar
                        )
                    )
                ]
                for candidate in sorted(
                    validation_passed,
                    key=lambda item: item.validation_metrics.calmar
                    if item.validation_metrics is not None
                    else -math.inf,
                    reverse=True,
                ):
                    signature = _candidate_signature(candidate)
                    if signature in objective_signatures:
                        continue
                    if not _is_feature_diverse(
                        candidate,
                        objective_candidates,
                        min_distance=config.min_feature_jaccard_distance,
                    ):
                        rejection_counts["feature_duplicate"] += 1
                        _append_jsonl(
                            rejected_path,
                            {
                                "candidate_id": candidate.candidate_id,
                                "reason": "feature_duplicate",
                                "train_calmar": candidate.train_metrics.calmar,
                                "validation_calmar": None
                                if candidate.validation_metrics is None
                                else candidate.validation_metrics.calmar,
                            },
                        )
                        continue
                    if config.reject_same_feature_family and _same_feature_family(
                        candidate,
                        objective_candidates,
                    ):
                        rejection_counts["feature_family_duplicate"] += 1
                        _append_jsonl(
                            rejected_path,
                            {
                                "candidate_id": candidate.candidate_id,
                                "reason": "feature_family_duplicate",
                                "train_calmar": candidate.train_metrics.calmar,
                                "validation_calmar": None
                                if candidate.validation_metrics is None
                                else candidate.validation_metrics.calmar,
                            },
                        )
                        continue
                    if not _is_behavior_diverse(
                        candidate,
                        objective_candidates,
                        min_distance=config.min_behavior_distance,
                    ):
                        rejection_counts["behavior_duplicate"] += 1
                        _append_jsonl(
                            rejected_path,
                            {
                                "candidate_id": candidate.candidate_id,
                                "reason": "behavior_duplicate",
                                "train_calmar": candidate.train_metrics.calmar,
                                "validation_calmar": None
                                if candidate.validation_metrics is None
                                else candidate.validation_metrics.calmar,
                            },
                        )
                        continue
                    objective_candidates.append(candidate)
                    objective_signatures.add(signature)
                    _append_jsonl(output_dir / "objective_candidates.jsonl", candidate.to_dict())
                    if candidate.candidate_id in rows_by_id:
                        _write_objective_artifact(
                            objective_artifacts_dir,
                            rows_by_id[candidate.candidate_id],
                            train_xy[2],
                        )
                    if len(objective_candidates) >= config.target_objective_count:
                        break
                best_train = top[0] if top else None
                objective_met = len(objective_candidates) >= config.target_objective_count
                _append_jsonl(
                    progress_path,
                    {
                        "event": "batch_completed",
                        "batch": batches_completed,
                        "candidates_evaluated": candidates_evaluated,
                        "best_train": None if best_train is None else best_train.to_dict(),
                        "best_validation": None
                        if best_validation is None
                        else best_validation.to_dict(),
                        "objective_candidates_found": len(objective_candidates),
                        "target_objective_count": config.target_objective_count,
                        "train_subperiod_count": config.train_subperiod_count,
                        "validation_subperiod_count": validation_subperiod_count,
                        "min_train_subperiod_calmar": config.min_train_subperiod_calmar,
                        "min_validation_subperiod_calmar": config.min_validation_subperiod_calmar,
                        "min_train_cagr": config.min_train_cagr,
                        "min_validation_cagr": config.min_validation_cagr,
                        "max_train_mdd": config.max_train_mdd,
                        "max_validation_mdd": config.max_validation_mdd,
                        "min_train_annual_return": config.min_train_annual_return,
                        "min_validation_annual_return": config.min_validation_annual_return,
                        "min_train_annual_calmar": config.min_train_annual_calmar,
                        "min_validation_annual_calmar": config.min_validation_annual_calmar,
                        "max_train_validation_calmar_ratio": config.max_train_validation_calmar_ratio,
                        "min_validation_excess_pvalue": config.min_validation_excess_pvalue,
                        "min_validation_bootstrap_calmar_p05": config.min_validation_bootstrap_calmar_p05,
                        "min_validation_bootstrap_excess_calmar_p05": config.min_validation_bootstrap_excess_calmar_p05,
                        "max_validation_random_baseline_pvalue": config.max_validation_random_baseline_pvalue,
                        "min_validation_deflated_sharpe": config.min_validation_deflated_sharpe,
                        "max_validation_pbo": config.max_validation_pbo,
                        "min_feature_ablation_validation_calmar": config.min_feature_ablation_validation_calmar,
                        "min_validation_regime_calmar": config.min_validation_regime_calmar,
                        "max_validation_trade_concentration": config.max_validation_trade_concentration,
                        "min_trades_per_year": config.min_trades_per_year,
                        "max_trades_per_year": config.max_trades_per_year,
                        "min_long_fraction": config.min_long_fraction,
                        "max_long_fraction": config.max_long_fraction,
                        "min_behavior_distance": config.min_behavior_distance,
                        "behavior_counts": dict(behavior_counts),
                        "rejection_counts": dict(rejection_counts),
                        "adaptive_family_search": config.adaptive_family_search,
                        "adaptive_quick_screen_candidates": config.adaptive_quick_screen_candidates,
                        "adaptive_family_attempts": dict(adaptive_family_attempts),
                        "adaptive_family_rewards": dict(adaptive_family_rewards),
                        "objective_met": objective_met,
                        "time_limit_reached": time_limit_reached,
                        "locked_opened": False,
                        "elapsed_seconds": time.perf_counter() - started,
                        "updated_at_utc": _now(),
                    },
                )
                _write_json(
                    status_path,
                    _status_payload(
                        config,
                        output_dir,
                        "objective_met" if objective_met else "running",
                        candidates_evaluated=candidates_evaluated,
                        batches_completed=batches_completed,
                        best_train=best_train,
                        best_validation=best_validation,
                        objective_candidates=tuple(objective_candidates),
                        objective_met=objective_met,
                        used_columns=tuple(frame.columns),
                        route_errors=tuple(route_errors),
                        behavior_counts=dict(behavior_counts),
                        rejection_counts=dict(rejection_counts),
                        adaptive_family_attempts=dict(adaptive_family_attempts),
                        adaptive_family_rewards=dict(adaptive_family_rewards),
                        elapsed_seconds=time.perf_counter() - started,
                    ),
                )
                if objective_met:
                    break
                if _time_limit_reached(started, config.time_limit_seconds):
                    time_limit_reached = True
                    _write_json(
                        status_path,
                        _status_payload(
                            config,
                            output_dir,
                            "time_limit",
                            candidates_evaluated=candidates_evaluated,
                            batches_completed=batches_completed,
                            best_train=best_train,
                            best_validation=best_validation,
                            objective_candidates=tuple(objective_candidates),
                            objective_met=False,
                            used_columns=tuple(frame.columns),
                            route_errors=tuple(route_errors),
                            behavior_counts=dict(behavior_counts),
                            rejection_counts=dict(rejection_counts),
                            adaptive_family_attempts=dict(adaptive_family_attempts),
                            adaptive_family_rewards=dict(adaptive_family_rewards),
                            elapsed_seconds=time.perf_counter() - started,
                        )
                        | {"time_limit_reached": True},
                    )
                    break

        if config.include_kronos and not objective_met:
            route_errors.extend(_run_kronos_challenger(config))

        report = MLSearchReport(
            status="objective_met"
            if objective_met
            else ("time_limit" if time_limit_reached else "completed"),
            locked_opened=False,
            objective_met=objective_met,
            run_id=config.run_id,
            output_dir=str(output_dir),
            symbol=config.symbol,
            workers=config.workers,
            candidates_evaluated=candidates_evaluated,
            batches_completed=batches_completed,
            train_period=_period_tuple(train),
            validation_period=_period_tuple(validation),
            locked_period=(config.locked_start, "closed"),
            used_columns=tuple(frame.columns),
            route_errors=tuple(route_errors),
            best_train=top[0] if top else None,
            best_validation=best_validation,
            objective_candidates=tuple(objective_candidates),
            top=tuple(top),
        )
        _write_json(best_path, report.to_dict())
        best_md_path.write_text(report_to_markdown(report), encoding="utf-8")
        _write_json(
            status_path,
            report.to_dict()
            | {
                "selection_phase": "train",
                "validation_used_for_selection": False,
                "validation_examined_after_train_pass": config.validation_target_calmar is not None,
                "target_objective_count": config.target_objective_count,
                "min_feature_jaccard_distance": config.min_feature_jaccard_distance,
                "min_behavior_distance": config.min_behavior_distance,
                "train_subperiod_count": config.train_subperiod_count,
                "validation_subperiod_count": validation_subperiod_count,
                "min_train_subperiod_calmar": config.min_train_subperiod_calmar,
                "min_validation_subperiod_calmar": config.min_validation_subperiod_calmar,
                "min_train_cagr": config.min_train_cagr,
                "min_validation_cagr": config.min_validation_cagr,
                "max_train_mdd": config.max_train_mdd,
                "max_validation_mdd": config.max_validation_mdd,
                "min_train_annual_return": config.min_train_annual_return,
                "min_validation_annual_return": config.min_validation_annual_return,
                "min_train_annual_calmar": config.min_train_annual_calmar,
                "min_validation_annual_calmar": config.min_validation_annual_calmar,
                "max_train_validation_calmar_ratio": config.max_train_validation_calmar_ratio,
                "min_validation_excess_pvalue": config.min_validation_excess_pvalue,
                "min_validation_bootstrap_calmar_p05": config.min_validation_bootstrap_calmar_p05,
                "min_validation_bootstrap_excess_calmar_p05": config.min_validation_bootstrap_excess_calmar_p05,
                "max_validation_random_baseline_pvalue": config.max_validation_random_baseline_pvalue,
                "min_validation_deflated_sharpe": config.min_validation_deflated_sharpe,
                "max_validation_pbo": config.max_validation_pbo,
                "min_feature_ablation_validation_calmar": config.min_feature_ablation_validation_calmar,
                "min_validation_regime_calmar": config.min_validation_regime_calmar,
                "max_validation_trade_concentration": config.max_validation_trade_concentration,
                "statistical_bootstrap_paths": config.statistical_bootstrap_paths,
                "statistical_bootstrap_block": config.statistical_bootstrap_block,
                "statistical_random_shuffles": config.statistical_random_shuffles,
                "statistical_pbo_splits": config.statistical_pbo_splits,
                "effective_dsr_trials": config.effective_dsr_trials,
                "time_limit_seconds": config.time_limit_seconds,
                "time_limit_reached": time_limit_reached,
                "defer_robustness_until_basic_pass": config.defer_robustness_until_basic_pass,
                "adaptive_family_search": config.adaptive_family_search,
                "adaptive_quick_screen_candidates": config.adaptive_quick_screen_candidates,
                "adaptive_family_min_weight": config.adaptive_family_min_weight,
                "adaptive_family_reward": config.adaptive_family_reward,
                "adaptive_initial_family_attempts": dict(config.adaptive_initial_family_attempts or {}),
                "adaptive_initial_family_rewards": dict(config.adaptive_initial_family_rewards or {}),
                "penalized_feature_pools": list(config.penalized_feature_pools),
                "penalized_feature_pool_factor": config.penalized_feature_pool_factor,
                "min_trades_per_year": config.min_trades_per_year,
                "max_trades_per_year": config.max_trades_per_year,
                "min_long_fraction": config.min_long_fraction,
                "max_long_fraction": config.max_long_fraction,
                "max_features_per_candidate": config.max_features_per_candidate,
                "include_pending_features": config.include_pending_features,
                "reject_same_feature_family": config.reject_same_feature_family,
                "pending_feature_library": config.pending_feature_library,
                "pending_feature_version": config.pending_feature_version,
                "behavior_counts": dict(behavior_counts),
                "rejection_counts": dict(rejection_counts),
                "adaptive_family_attempts": dict(adaptive_family_attempts),
                "adaptive_family_rewards": dict(adaptive_family_rewards),
                "elapsed_seconds": time.perf_counter() - started,
                "completed_at_utc": _now(),
            },
        )
        return report
    except Exception as exc:
        stderr_path.write_text(traceback.format_exc(), encoding="utf-8")
        _write_json(
            status_path,
            {
                "status": "error",
                "locked_opened": False,
                "objective_met": False,
                "run_id": config.run_id,
                "error": str(exc),
                "updated_at_utc": _now(),
            },
        )
        raise


def load_ml_frame(
    symbol: str = "SPY",
    *,
    library: str = "prices_daily",
    end: str | None = None,
) -> pd.DataFrame:
    store = TimeSeriesStore(base_data_dir() / "timeseries")
    source = store.read(library=library, symbol=symbol, end=end)
    columns = {str(column) for column in source.columns}
    forbidden = [column for column in columns if any(token in column.lower() for token in FORBIDDEN_LOCKED_COLUMNS)]
    if forbidden:
        raise ValueError(f"ml-search source has forbidden columns: {sorted(forbidden)}")
    unknown = columns - ALLOWED_SOURCE_COLUMNS
    if unknown:
        raise ValueError(f"ml-search source has unsupported columns: {sorted(unknown)}")
    missing = REQUIRED_SOURCE_COLUMNS - columns
    if missing:
        raise ValueError(f"ml-search source missing columns: {sorted(missing)}")

    wanted = ["open", "high", "low", "close", "adj_close", *(["volume"] if "volume" in columns else [])]
    frame = source[wanted].copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "adj_close"])
    ratio = frame["adj_close"] / frame["close"]
    adjusted = pd.DataFrame(index=frame.index)
    for column in OHLC_COLUMNS:
        adjusted[column] = frame[column] * ratio
    if "volume" in frame.columns:
        adjusted["volume"] = frame["volume"].fillna(0.0)
    return adjusted.dropna()


def load_pending_feature_frame(
    symbol: str = "SPY",
    *,
    library: str = "features_pending_daily",
    version: str = "pending_features_v1",
    end: str | None = None,
) -> pd.DataFrame:
    """Load the materialized pending feature panel without touching locked data."""

    store = TimeSeriesStore(base_data_dir() / "timeseries")
    try:
        frame = store.read(library, symbol, version=version, end=end)
    except Exception as exc:
        raise ValueError(
            "pending features are not available yet; run "
            "python -m aurora.research.pending_features --version "
            f"{version}"
        ) from exc

    columns = [str(column) for column in frame.columns]
    forbidden = [
        column
        for column in columns
        if any(token in column.lower() for token in FORBIDDEN_LOCKED_COLUMNS)
    ]
    if forbidden:
        raise ValueError(
            f"pending feature panel has forbidden columns: {sorted(forbidden)}"
        )

    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def join_pending_features(
    base_features: pd.DataFrame,
    pending_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach external/pending features with a prefix to avoid name collisions."""

    pending = pending_features.add_prefix("pending_")
    pending = pending.reindex(base_features.index).ffill()
    joined = pd.concat([base_features, pending], axis=1)
    return joined.replace([np.inf, -np.inf], np.nan)


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cfg = FeaturePipelineConfig(
        rolling_windows=(5, 10, 20, 60, 126),
        return_lags=(1, 2, 3, 5, 10, 21),
        include_technicals=True,
        include_microstructure="volume" in frame.columns,
        include_volatility=True,
        standardize=True,
        standardize_method="rolling_zscore",
        standardize_window=252,
        price_col="close",
    )
    pipeline = FeaturePipeline(cfg)
    base = pipeline.fit_transform(frame)
    close = frame["close"].astype(float)
    candle = pd.DataFrame(index=frame.index)
    candle["pa_gap"] = frame["open"] / close.shift(1) - 1.0
    candle["pa_body"] = frame["close"] / frame["open"] - 1.0
    candle["pa_range"] = frame["high"] / frame["low"] - 1.0
    candle["pa_close_pos"] = (frame["close"] - frame["low"]) / (frame["high"] - frame["low"]).replace(0.0, np.nan)
    candle["pa_overnight"] = frame["open"] / close.shift(1) - 1.0
    out = pd.concat([base, candle], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(how="all")


def positions_from_scores(scores: np.ndarray, *, threshold: float, direction: int) -> np.ndarray:
    signed = np.asarray(scores, dtype=np.float64) * float(direction)
    return np.where(signed >= float(threshold), 1.0, -1.0)


def metrics_from_next_returns(next_returns: np.ndarray, positions: np.ndarray) -> MLSearchMetrics:
    returns = np.asarray(next_returns, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    strategy_returns = pos * returns
    equity = np.cumprod(1.0 + strategy_returns)
    final = float(equity[-1]) if len(equity) else 1.0
    years = _years_from_rows(len(strategy_returns))
    cagr = final ** (1.0 / years) - 1.0 if final > 0.0 else -1.0
    peak = np.maximum.accumulate(equity) if len(equity) else np.asarray([1.0])
    drawdown = equity / peak - 1.0 if len(equity) else np.asarray([0.0])
    mdd = float(np.min(drawdown)) if len(drawdown) else 0.0
    calmar = cagr / abs(mdd) if abs(mdd) > 1e-12 else (999.0 if cagr > 0 else 0.0)
    trades = int(np.sum(np.abs(np.diff(pos)) > 0.0)) if len(pos) > 1 else 0
    return MLSearchMetrics(
        calmar=float(calmar),
        cagr=float(cagr),
        mdd=float(mdd),
        trades=trades,
        trades_per_year=float(trades / years),
        long_fraction=float(np.mean(pos > 0.0)) if len(pos) else 0.0,
        final_nav=final,
    )


def _horizon_returns(next_returns: np.ndarray, horizon: int) -> np.ndarray:
    returns = np.asarray(next_returns, dtype=np.float64)
    h = max(1, int(horizon))
    out = np.full(len(returns), np.nan, dtype=np.float64)
    if len(returns) < h:
        return out
    for idx in range(0, len(returns) - h + 1):
        out[idx] = float(np.prod(1.0 + returns[idx: idx + h]) - 1.0)
    return out


def _target_values(horizon_returns: np.ndarray, target_type: str) -> np.ndarray:
    values = np.asarray(horizon_returns, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return values
    if target_type == "strong_return":
        return values - float(np.quantile(finite, 0.60))
    if target_type == "avoid_drawdown":
        return values - float(np.quantile(finite, 0.30))
    return values


def _smooth_scores(scores: np.ndarray, *, smoothing: int) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    window = max(1, int(smoothing))
    if window <= 1 or len(arr) < window:
        return arr
    return (
        pd.Series(arr)
        .rolling(window=window, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def _positions_signature(positions: np.ndarray, *, samples: int = 256) -> str:
    pos = np.asarray(positions, dtype=np.float64)
    if len(pos) <= samples:
        chosen = pos
    else:
        idx = np.linspace(0, len(pos) - 1, samples).round().astype(int)
        chosen = pos[idx]
    return "".join("1" if value > 0.0 else "0" for value in chosen)


def _behavior_vector(metrics: MLSearchMetrics) -> tuple[float, float, float]:
    return (
        float(metrics.long_fraction),
        float(metrics.trades_per_year),
        float(metrics.final_nav),
    )


def _behavior_bucket(metrics: MLSearchMetrics) -> str:
    if metrics.long_fraction < 0.02:
        return "always_short"
    if metrics.long_fraction > 0.98:
        return "always_long"
    if metrics.long_fraction < 0.40:
        return "short_bias"
    if metrics.long_fraction > 0.60:
        return "long_bias"
    return "balanced"


def _behavior_rejection_reason(
    metrics: MLSearchMetrics,
    *,
    min_trades_per_year: float,
    max_trades_per_year: float | None = None,
    min_long_fraction: float | None = None,
    max_long_fraction: float | None = None,
    always_long_threshold: float = 0.98,
    always_short_threshold: float = 0.02,
) -> str | None:
    if min_long_fraction is not None and metrics.long_fraction < float(min_long_fraction):
        return "too_short_biased"
    if max_long_fraction is not None and metrics.long_fraction > float(max_long_fraction):
        return "too_long_biased"
    if metrics.long_fraction > float(always_long_threshold):
        return "always_long"
    if metrics.long_fraction < float(always_short_threshold):
        return "always_short"
    if metrics.trades_per_year < float(min_trades_per_year):
        return "too_few_trades"
    if max_trades_per_year is not None and metrics.trades_per_year > float(max_trades_per_year):
        return "too_many_trades"
    return None


def report_to_markdown(report: MLSearchReport) -> str:
    lines = [
        "# Aurora ML Search",
        "",
        f"Run ID: `{report.run_id}`",
        f"Status: `{report.status}`",
        f"Locked opened: `{report.locked_opened}`",
        f"Objective met: `{report.objective_met}`",
        f"Workers: `{report.workers}`",
        f"Candidates evaluated: `{report.candidates_evaluated}`",
        f"Objective candidates found: `{len(report.objective_candidates)}`",
        "Train subperiod rule: Calmar >= 0.000 in each of 4 train subperiods",
        "",
        "| Rank | Candidate | Route | Model | Train Calmar | Valid Calmar | CAGR | MDD | Trades/year | Rule |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, candidate in enumerate(report.top, start=1):
        train = candidate.train_metrics
        valid = candidate.validation_metrics
        lines.append(
            f"| {rank} | {candidate.candidate_id} | {candidate.route} | {candidate.model} | "
            f"{train.calmar:.3f} | {'' if valid is None else f'{valid.calmar:.3f}'} | "
            f"{train.cagr * 100:.2f}% | {train.mdd * 100:.2f}% | "
            f"{train.trades_per_year:.1f} | {candidate.rule} |"
        )
    if report.objective_candidates:
        lines.extend(
            [
                "",
                "## Objective Candidates",
                "",
                "| Rank | Candidate | Train Calmar | Validation Calmar | Min train subperiod | Features |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for rank, candidate in enumerate(report.objective_candidates, start=1):
            valid = candidate.validation_metrics
            min_subperiod = min(
                (metrics.calmar for metrics in candidate.train_subperiod_metrics),
                default=float("nan"),
            )
            lines.append(
                f"| {rank} | {candidate.candidate_id} | "
                f"{candidate.train_metrics.calmar:.3f} | "
                f"{'' if valid is None else f'{valid.calmar:.3f}'} | "
                f"{min_subperiod:.3f} | "
                f"{len(candidate.feature_set)} |"
            )
    if report.route_errors:
        lines.extend(["", "## Route errors", ""])
        lines.extend(f"- {error}" for error in report.route_errors)
    return "\n".join(lines) + "\n"


def _candidate_signature(candidate: MLSearchCandidate) -> str:
    return json.dumps(
        {
            "route": candidate.route,
            "model": candidate.model,
            "features": sorted(candidate.feature_set),
            "threshold": round(float(candidate.threshold), 8),
            "direction": int(candidate.direction),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _is_feature_diverse(
    candidate: MLSearchCandidate,
    accepted: list[MLSearchCandidate],
    *,
    min_distance: float,
) -> bool:
    if not accepted:
        return True
    required = max(0.0, min(1.0, float(min_distance)))
    candidate_features = set(candidate.feature_set)
    for other in accepted:
        other_features = set(other.feature_set)
        union = candidate_features | other_features
        if not union:
            return False
        similarity = len(candidate_features & other_features) / len(union)
        distance = 1.0 - similarity
        if distance < required:
            return False
    return True


def _same_feature_family(
    candidate: MLSearchCandidate,
    accepted: list[MLSearchCandidate],
) -> bool:
    if not accepted:
        return False
    signature = _feature_family_signature(candidate.feature_set)
    return any(signature == _feature_family_signature(other.feature_set) for other in accepted)


def _feature_family_signature(features: tuple[str, ...]) -> tuple[str, ...]:
    families = sorted({_feature_family_name(name) for name in features})
    return tuple(families)


def _feature_family_name(name: str) -> str:
    lower = name.lower()
    if lower.startswith("pending_"):
        if any(token in lower for token in ("vix", "vvix", "skew")):
            return "pending_volatility_index"
        if any(token in lower for token in ("qqq", "iwm", "rsp", "efa", "eem")):
            return "pending_equity_market"
        if any(token in lower for token in ("gld", "uso", "hyg", "lqd")):
            return "pending_cross_asset"
        if any(token in lower for token in ("xle", "xlf", "xlk", "xlu", "xlp", "xly")):
            return "pending_sector"
        if "calendar" in lower:
            return "pending_calendar"
        if any(token in lower for token in ("gap", "range", "overnight", "intraday")):
            return "pending_price_structure"
        return "pending_other"
    if lower.startswith("pa_"):
        return "price_action"
    if lower.startswith(("rsi", "macd", "bb_")):
        return "technical"
    if "vol" in lower or "std" in lower:
        return "volatility"
    if "ret" in lower or "lag" in lower:
        return "returns"
    if lower.startswith("roll_"):
        return "rolling"
    return lower.split("_", 1)[0]


def _is_behavior_diverse(
    candidate: MLSearchCandidate,
    accepted: list[MLSearchCandidate],
    *,
    min_distance: float,
) -> bool:
    if not accepted:
        return True
    required = max(0.0, float(min_distance))
    for other in accepted:
        if candidate.positions_signature and candidate.positions_signature == other.positions_signature:
            return False
        if _behavior_distance(candidate.behavior_vector, other.behavior_vector) < required:
            return False
    return True


def _behavior_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) < 3 or len(right) < 3:
        return 0.0
    long_distance = abs(float(left[0]) - float(right[0]))
    trade_distance = abs(float(left[1]) - float(right[1])) / 150.0
    nav_distance = abs(
        math.log(max(float(left[2]), 1e-9)) - math.log(max(float(right[2]), 1e-9))
    ) / 5.0
    return float(long_distance + trade_distance + nav_distance)


def _feature_importance(model: Any) -> dict[str, float]:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    total = float(np.sum(np.abs(arr)))
    if total <= 0.0:
        return {str(idx): float(value) for idx, value in enumerate(arr)}
    return {str(idx): float(value / total) for idx, value in enumerate(arr)}


def _rejected_payload(row: dict[str, Any], feature_names: tuple[str, ...]) -> dict[str, Any]:
    spec = row["spec"]
    train = row.get("train_metrics", {})
    validation = row.get("validation_metrics")
    return {
        "candidate_id": spec.get("candidate_id"),
        "reason": row.get("rejection_reason"),
        "model": spec.get("model"),
        "horizon": spec.get("horizon"),
        "target_type": spec.get("target_type"),
        "threshold": spec.get("threshold"),
        "direction": spec.get("direction"),
        "features": [feature_names[int(idx)] for idx in spec.get("columns", [])],
        "train_calmar": train.get("calmar"),
        "train_cagr": train.get("cagr"),
        "train_mdd": train.get("mdd"),
        "train_long_fraction": train.get("long_fraction"),
        "train_trades_per_year": train.get("trades_per_year"),
        "validation_calmar": None if validation is None else validation.get("calmar"),
        "robustness": dict(row.get("robustness", {})),
    }


def _anti_overfit_ledger_payload(
    row: dict[str, Any],
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    spec = row["spec"]
    train = row.get("train_metrics", {})
    validation = row.get("validation_metrics")
    return {
        "candidate_id": spec.get("candidate_id"),
        "family": spec.get("feature_pool"),
        "model": spec.get("model"),
        "horizon": spec.get("horizon"),
        "threshold": spec.get("threshold"),
        "feature_set": [feature_names[int(idx)] for idx in spec.get("columns", [])],
        "train_calmar": train.get("calmar"),
        "validation_calmar": None if validation is None else validation.get("calmar"),
        "reason": row.get("rejection_reason"),
        "failed_locked": False,
        "positions_signature": row.get("positions_signature", ""),
    }


def _load_anti_overfit_patterns(path: str | None) -> tuple[dict[str, Any], ...]:
    if not path:
        return tuple()
    ledger = Path(path)
    if not ledger.exists():
        return tuple()
    patterns: list[dict[str, Any]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        features = tuple(str(value) for value in item.get("feature_set", item.get("features", ())))
        if not features:
            continue
        patterns.append(
            {
                "feature_set": features,
                "model": str(item.get("model", "")),
                "family": str(item.get("family", "")),
                "failed_locked": bool(item.get("failed_locked", False)),
            }
        )
    return tuple(patterns)


def _anti_overfit_penalty(
    feature_set: tuple[str, ...],
    *,
    model: str,
    family: str,
    patterns: tuple[dict[str, Any], ...],
) -> float:
    if not patterns:
        return 0.0
    current = set(str(value) for value in feature_set)
    if not current:
        return 0.0
    penalty = 0.0
    for pattern in patterns:
        previous = set(str(value) for value in pattern.get("feature_set", ()))
        if not previous:
            continue
        overlap = len(current & previous) / float(len(current | previous))
        if overlap < 0.50:
            continue
        local = overlap
        if str(pattern.get("model", "")) == str(model):
            local += 0.25
        if str(pattern.get("family", "")) == str(family):
            local += 0.25
        if bool(pattern.get("failed_locked", False)):
            local += 0.50
        penalty = max(penalty, local)
    return float(penalty)


def _survival_score_from_parts(
    train_metrics: MLSearchMetrics,
    validation_metrics: MLSearchMetrics | None,
    robustness: dict[str, Any],
    *,
    complexity_penalty: float,
    repeated_failure_penalty: float,
) -> float:
    valid_calmar = (
        -10.0 if validation_metrics is None else float(validation_metrics.calmar)
    )
    train_calmar = float(train_metrics.calmar)
    gap_penalty = max(0.0, train_calmar - max(valid_calmar, 0.0) * 2.0)
    bootstrap = float(robustness.get("bootstrap_calmar_p05", 0.0) or 0.0)
    instability = 0.0
    if validation_metrics is not None and validation_metrics.mdd < -0.30:
        instability += abs(float(validation_metrics.mdd)) - 0.30
    return float(
        valid_calmar
        + bootstrap
        - 0.25 * gap_penalty
        - float(complexity_penalty)
        - float(repeated_failure_penalty)
        - instability
    )


def _importance_payload(row: dict[str, Any], feature_names: tuple[str, ...]) -> dict[str, Any]:
    spec = row["spec"]
    local_features = [feature_names[int(idx)] for idx in spec.get("columns", [])]
    importance = {
        local_features[int(idx)]: float(value)
        for idx, value in row.get("feature_importance", {}).items()
        if int(idx) < len(local_features)
    }
    return {
        "candidate_id": spec.get("candidate_id"),
        "model": spec.get("model"),
        "importance": importance,
    }


def _write_objective_artifact(
    output_dir: Path,
    row: dict[str, Any],
    feature_names: tuple[str, ...],
) -> None:
    spec = row["spec"]
    candidate_id = str(spec.get("candidate_id", "candidate"))
    feature_set = [feature_names[int(idx)] for idx in spec.get("columns", [])]
    payload = {
        "candidate_id": candidate_id,
        "spec": dict(spec),
        "features": feature_set,
        "feature_family_signature": list(_feature_family_signature(tuple(feature_set))),
        "seed": spec.get("seed"),
        "train_metrics": row.get("train_metrics"),
        "validation_metrics": row.get("validation_metrics"),
        "train_subperiod_metrics": row.get("train_subperiod_metrics", []),
        "validation_subperiod_metrics": row.get("validation_subperiod_metrics", []),
        "robustness": dict(row.get("robustness", {})),
        "positions_signature": row.get("positions_signature", ""),
        "behavior_vector": list(row.get("behavior_vector", ())),
        "behavior_bucket": row.get("behavior_bucket"),
        "feature_importance": _importance_payload(row, feature_names).get("importance", {}),
        "locked_opened": False,
        "created_at_utc": _now(),
    }
    _write_json(output_dir / f"{candidate_id}.json", payload)


def _init_worker(
    x_train: np.ndarray,
    y_train: np.ndarray,
    ret_train_next: np.ndarray,
    x_valid: np.ndarray,
    ret_valid_next: np.ndarray,
    train_subperiod_masks: tuple[np.ndarray, ...],
    train_annual_return_masks: tuple[np.ndarray, ...],
    valid_subperiod_masks: tuple[np.ndarray, ...],
    valid_annual_return_masks: tuple[np.ndarray, ...],
    train_years: float,
    valid_years: float,
) -> None:
    global _X_TRAIN, _Y_TRAIN, _RET_TRAIN_NEXT, _X_VALID, _RET_VALID_NEXT, _TRAIN_SUBPERIOD_MASKS, _TRAIN_ANNUAL_RETURN_MASKS, _VALID_SUBPERIOD_MASKS, _VALID_ANNUAL_RETURN_MASKS, _TRAIN_YEARS, _VALID_YEARS
    _X_TRAIN = x_train
    _Y_TRAIN = y_train
    _RET_TRAIN_NEXT = ret_train_next
    _X_VALID = x_valid
    _RET_VALID_NEXT = ret_valid_next
    _TRAIN_SUBPERIOD_MASKS = train_subperiod_masks
    _TRAIN_ANNUAL_RETURN_MASKS = train_annual_return_masks
    _VALID_SUBPERIOD_MASKS = valid_subperiod_masks
    _VALID_ANNUAL_RETURN_MASKS = valid_annual_return_masks
    _TRAIN_YEARS = train_years
    _VALID_YEARS = valid_years


def _evaluate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    assert _X_TRAIN is not None
    assert _Y_TRAIN is not None
    assert _RET_TRAIN_NEXT is not None
    cols = np.asarray(spec["columns"], dtype=int)
    x_train = _fill_matrix(_X_TRAIN[:, cols])
    train_payload = _fit_predict_payload(spec, x_train, _Y_TRAIN, x_train)
    train_scores = _smooth_scores(
        train_payload["train_scores"],
        smoothing=int(spec.get("smoothing", 1)),
    )
    train_positions = positions_from_scores(
        train_scores,
        threshold=float(spec["threshold"]),
        direction=int(spec["direction"]),
    )
    train_metrics = metrics_from_next_returns(_RET_TRAIN_NEXT, train_positions)
    behavior_reason = _behavior_rejection_reason(
        train_metrics,
        min_trades_per_year=float(spec["min_trades_per_year"]),
        max_trades_per_year=spec.get("max_trades_per_year"),
        min_long_fraction=spec.get("min_long_fraction"),
        max_long_fraction=spec.get("max_long_fraction"),
        always_long_threshold=float(spec["always_long_threshold"]),
        always_short_threshold=float(spec["always_short_threshold"]),
    )
    train_subperiod_metrics = tuple(
        metrics_from_next_returns(_RET_TRAIN_NEXT[mask], train_positions[mask])
        for mask in _TRAIN_SUBPERIOD_MASKS
        if bool(np.any(mask))
    )
    subperiod_ok = all(
        metrics.calmar >= float(spec["min_train_subperiod_calmar"])
        for metrics in train_subperiod_metrics
    )
    min_train_annual_return = spec.get("min_train_annual_return")
    train_annual_ok = _train_annual_return_ok(
        _RET_TRAIN_NEXT,
        train_positions,
        min_return=min_train_annual_return,
    )
    train_annual_calmar_ok = _period_annual_calmar_ok(
        _RET_TRAIN_NEXT,
        train_positions,
        _TRAIN_ANNUAL_RETURN_MASKS,
        min_calmar=spec.get("min_train_annual_calmar"),
    )
    train_total_ok = _metrics_total_ok(
        train_metrics,
        min_cagr=spec.get("min_train_cagr"),
        max_mdd=spec.get("max_train_mdd"),
    )
    train_calmar_ceiling_reason = _train_calmar_ceiling_reason(
        train_metrics,
        max_train_calmar=spec.get("max_train_calmar"),
    )
    valid_metrics = None
    valid_positions: np.ndarray | None = None
    valid_subperiod_metrics: tuple[MLSearchMetrics, ...] = tuple()
    robustness: dict[str, Any] = {}
    rejection_reason = None
    if train_metrics.calmar < float(spec["target_calmar"]):
        rejection_reason = "train_calmar"
    elif train_calmar_ceiling_reason is not None:
        rejection_reason = train_calmar_ceiling_reason
    elif not train_total_ok:
        rejection_reason = "train_total_metrics"
    elif not subperiod_ok:
        rejection_reason = "train_subperiod_calmar"
    elif not train_annual_ok:
        rejection_reason = "train_annual_return"
    elif not train_annual_calmar_ok:
        rejection_reason = "train_annual_calmar"
    elif behavior_reason is not None:
        rejection_reason = behavior_reason
    if (
        train_metrics.calmar >= float(spec["target_calmar"])
        and train_total_ok
        and train_calmar_ceiling_reason is None
        and subperiod_ok
        and train_annual_ok
        and train_annual_calmar_ok
        and behavior_reason is None
        and _X_VALID is not None
        and _RET_VALID_NEXT is not None
    ):
        x_valid = _fill_matrix(_X_VALID[:, cols])
        valid_payload = _fit_predict_payload(spec, x_train, _Y_TRAIN, x_train, x_valid)
        valid_scores = _smooth_scores(
            valid_payload["validation_scores"],
            smoothing=int(spec.get("smoothing", 1)),
        )
        valid_positions = positions_from_scores(
            valid_scores,
            threshold=float(spec["threshold"]),
            direction=int(spec["direction"]),
        )
        valid_metrics = metrics_from_next_returns(_RET_VALID_NEXT, valid_positions)
        valid_behavior_reason = _behavior_rejection_reason(
            valid_metrics,
            min_trades_per_year=float(spec["min_trades_per_year"]),
            max_trades_per_year=spec.get("max_trades_per_year"),
            min_long_fraction=spec.get("min_long_fraction"),
            max_long_fraction=spec.get("max_long_fraction"),
            always_long_threshold=float(spec["always_long_threshold"]),
            always_short_threshold=float(spec["always_short_threshold"]),
        )
        valid_subperiod_metrics = tuple(
            metrics_from_next_returns(_RET_VALID_NEXT[mask], valid_positions[mask])
            for mask in _VALID_SUBPERIOD_MASKS
            if bool(np.any(mask))
        )
        min_validation_subperiod_calmar = spec.get("min_validation_subperiod_calmar")
        validation_subperiod_ok = (
            True
            if min_validation_subperiod_calmar is None
            else all(
                metrics.calmar >= float(min_validation_subperiod_calmar)
                for metrics in valid_subperiod_metrics
            )
        )
        validation_annual_ok = _period_annual_return_ok(
            _RET_VALID_NEXT,
            valid_positions,
            _VALID_ANNUAL_RETURN_MASKS,
            min_return=spec.get("min_validation_annual_return"),
        )
        validation_annual_calmar_ok = _period_annual_calmar_ok(
            _RET_VALID_NEXT,
            valid_positions,
            _VALID_ANNUAL_RETURN_MASKS,
            min_calmar=spec.get("min_validation_annual_calmar"),
        )
        validation_total_ok = _metrics_total_ok(
            valid_metrics,
            min_cagr=spec.get("min_validation_cagr"),
            max_mdd=spec.get("max_validation_mdd"),
        )
        ratio_ok = _train_validation_ratio_ok(
            train_metrics,
            valid_metrics,
            max_ratio=spec.get("max_train_validation_calmar_ratio"),
        )
        validation_target = spec.get("validation_target_calmar")
        basic_validation_reason = None
        if validation_target is not None and valid_metrics.calmar < float(validation_target):
            basic_validation_reason = "validation_calmar"
        elif not validation_total_ok:
            basic_validation_reason = "validation_total_metrics"
        elif not validation_subperiod_ok:
            basic_validation_reason = "validation_subperiod_calmar"
        elif not validation_annual_ok:
            basic_validation_reason = "validation_annual_return"
        elif not validation_annual_calmar_ok:
            basic_validation_reason = "validation_annual_calmar"
        elif not ratio_ok:
            basic_validation_reason = "train_validation_calmar_ratio"
        elif valid_behavior_reason is not None:
            basic_validation_reason = f"validation_{valid_behavior_reason}"
        if basic_validation_reason is not None and bool(
            spec.get("defer_robustness_until_basic_pass", False)
        ):
            rejection_reason = basic_validation_reason
        else:
            robustness = _validation_robustness_summary(
                _RET_VALID_NEXT,
                valid_positions,
                paths=int(spec.get("statistical_bootstrap_paths", 300)),
                block=int(spec.get("statistical_bootstrap_block", 21)),
                seed=int(spec.get("seed", 42)),
                n_trials=int(spec.get("effective_dsr_trials") or spec.get("n_trials", 1)),
                random_shuffles=int(spec.get("statistical_random_shuffles", 300)),
                pbo_splits=int(spec.get("statistical_pbo_splits", 8)),
            )
            ablation_threshold = spec.get("min_feature_ablation_validation_calmar")
            if ablation_threshold is not None:
                robustness["feature_ablation_validation_calmar"] = _feature_ablation_calmar(
                    spec,
                    cols,
                    train_payload.get("feature_importance", {}),
                    validation_target_shape=len(valid_positions),
                )
            statistical_ok = _validation_statistical_ok(
                robustness,
                max_excess_pvalue=spec.get("min_validation_excess_pvalue"),
                min_bootstrap_calmar_p05=spec.get("min_validation_bootstrap_calmar_p05"),
                min_bootstrap_excess_calmar_p05=spec.get(
                    "min_validation_bootstrap_excess_calmar_p05"
                ),
                max_random_baseline_pvalue=spec.get("max_validation_random_baseline_pvalue"),
                min_deflated_sharpe=spec.get("min_validation_deflated_sharpe"),
                max_pbo=spec.get("max_validation_pbo"),
                min_feature_ablation_validation_calmar=ablation_threshold,
                min_regime_calmar=spec.get("min_validation_regime_calmar"),
                max_trade_concentration=spec.get("max_validation_trade_concentration"),
            )
            if basic_validation_reason is not None:
                rejection_reason = basic_validation_reason
            elif statistical_ok is not None:
                rejection_reason = statistical_ok
    return {
        "candidate": True,
        "spec": spec,
        "train_metrics": train_metrics.to_dict(),
        "train_subperiod_metrics": [metrics.to_dict() for metrics in train_subperiod_metrics],
        "validation_subperiod_metrics": [
            metrics.to_dict() for metrics in valid_subperiod_metrics
        ],
        "train_subperiod_ok": subperiod_ok,
        "train_annual_ok": train_annual_ok,
        "train_annual_calmar_ok": train_annual_calmar_ok,
        "validation_metrics": None if valid_metrics is None else valid_metrics.to_dict(),
        "positions_signature": _positions_signature(train_positions),
        "behavior_vector": _behavior_vector(train_metrics),
        "behavior_bucket": _behavior_bucket(train_metrics),
        "rejection_reason": rejection_reason,
        "feature_importance": train_payload.get("feature_importance", {}),
        "robustness": robustness,
        "survival_score": _survival_score_from_parts(
            train_metrics,
            valid_metrics,
            robustness,
            complexity_penalty=float(spec.get("complexity_penalty", 0.0)),
            repeated_failure_penalty=float(spec.get("repeated_failure_penalty", 0.0)),
        ),
    }


def _fit_predict_payload(
    spec: dict[str, Any],
    x_train: np.ndarray,
    train_next_returns: np.ndarray,
    x_train_predict: np.ndarray,
    x_valid_predict: np.ndarray | None = None,
) -> dict[str, Any]:
    horizon_returns = _horizon_returns(train_next_returns, int(spec.get("horizon", 1)))
    fit_mask = np.isfinite(horizon_returns)
    x_fit = x_train[fit_mask]
    y_fit = _target_values(horizon_returns[fit_mask], str(spec.get("target_type", "direction")))
    model = _fit_model(spec, x_fit, y_fit)
    train_scores = _predict_scores(model, str(spec["model"]), x_train_predict)
    payload: dict[str, Any] = {
        "train_scores": train_scores,
        "feature_importance": _feature_importance(model),
    }
    if x_valid_predict is not None:
        payload["validation_scores"] = _predict_scores(
            model,
            str(spec["model"]),
            x_valid_predict,
        )
    return payload


def _fit_predict_scores(
    spec: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
) -> np.ndarray:
    return _fit_predict_payload(spec, x_train, y_train, x_predict)["train_scores"]


def _fit_model(spec: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray) -> Any:
    model = str(spec["model"])
    if model == "corr":
        return {"kind": "corr", "weights": _corr_weights(x_train, y_train)}
    if model == "ridge":
        return {
            "kind": "ridge",
            "weights": _ridge_weights(x_train, y_train, alpha=float(spec.get("alpha", 1.0))),
        }
    if model == "logistic":
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.exceptions import ConvergenceWarning
        except Exception:
            return {"kind": "corr", "weights": _corr_weights(x_train, y_train)}
        labels = (y_train > 0.0).astype(int)
        if len(set(labels.tolist())) < 2:
            return {"kind": "corr", "weights": _corr_weights(x_train, y_train)}
        clf = LogisticRegression(max_iter=300, C=float(spec.get("c", 1.0)), random_state=int(spec["seed"]))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf.fit(x_train, labels)
        return clf
    if model == "forest":
        try:
            from sklearn.ensemble import RandomForestClassifier
        except Exception:
            return {"kind": "corr", "weights": _corr_weights(x_train, y_train)}
        labels = (y_train > 0.0).astype(int)
        if len(set(labels.tolist())) < 2:
            return {"kind": "corr", "weights": _corr_weights(x_train, y_train)}
        clf = RandomForestClassifier(
            n_estimators=int(spec.get("n_estimators", 80)),
            max_depth=int(spec.get("max_depth", 3)),
            min_samples_leaf=int(spec.get("min_samples_leaf", 20)),
            random_state=int(spec["seed"]),
            n_jobs=1,
        )
        clf.fit(x_train, labels)
        return clf
    if model == "lightgbm":
        return _fit_lightgbm(spec, x_train, y_train)
    if model == "xgboost":
        return _fit_xgboost(spec, x_train, y_train)
    return {"kind": "corr", "weights": _corr_weights(x_train, y_train)}


def _fit_lightgbm(spec: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray) -> Any:
    try:
        module = importlib.import_module("lightgbm")
    except Exception as exc:
        raise RuntimeError("Missing optional dependency for lightgbm. Install with: pip install lightgbm") from exc
    labels = (y_train > 0.0).astype(int)
    if len(set(labels.tolist())) < 2:
        return {"kind": "corr", "weights": _corr_weights(x_train, y_train)}
    params = dict(spec.get("params", {}))
    clf = module.LGBMClassifier(
        objective="binary",
        random_state=int(spec["seed"]),
        n_jobs=1,
        verbosity=-1,
        **params,
    )
    clf.fit(x_train, labels)
    return clf


def _fit_xgboost(spec: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray) -> Any:
    try:
        module = importlib.import_module("xgboost")
    except Exception as exc:
        raise RuntimeError("Missing optional dependency for xgboost. Install with: pip install xgboost") from exc
    labels = (y_train > 0.0).astype(int)
    if len(set(labels.tolist())) < 2:
        return {"kind": "corr", "weights": _corr_weights(x_train, y_train)}
    params = dict(spec.get("params", {}))
    clf = module.XGBClassifier(
        objective="binary:logistic",
        random_state=int(spec["seed"]),
        n_jobs=1,
        eval_metric="logloss",
        verbosity=0,
        **params,
    )
    clf.fit(x_train, labels)
    return clf


def _predict_scores(model: Any, model_name: str, x_predict: np.ndarray) -> np.ndarray:
    if isinstance(model, dict) and model.get("kind") in {"corr", "ridge"}:
        return np.asarray(x_predict, dtype=np.float64) @ np.asarray(model["weights"], dtype=np.float64)
    if hasattr(model, "predict_proba"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            proba = model.predict_proba(x_predict)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError(f"{model_name} did not return binary probabilities")
        return np.asarray(proba[:, 1], dtype=np.float64) - 0.5
    return np.asarray(model.predict(x_predict), dtype=np.float64)


def _corr_scores(x_train: np.ndarray, y_train: np.ndarray, x_predict: np.ndarray) -> np.ndarray:
    return np.asarray(x_predict, dtype=np.float64) @ _corr_weights(x_train, y_train)


def _corr_weights(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    y = np.asarray(y_train, dtype=np.float64)
    x = np.asarray(x_train, dtype=np.float64)
    denom = np.sqrt(np.sum(x * x, axis=0)) * max(float(np.sqrt(np.sum(y * y))), 1e-12)
    numerator = np.sum(x * y[:, None], axis=0)
    weights = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denom, out=weights, where=denom > 0.0)
    return np.asarray(weights, dtype=np.float64)


def _ridge_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    return np.asarray(x_predict, dtype=np.float64) @ _ridge_weights(x_train, y_train, alpha=alpha)


def _ridge_weights(x_train: np.ndarray, y_train: np.ndarray, *, alpha: float) -> np.ndarray:
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    xtx = x.T @ x
    ridge = max(float(alpha), 1e-9) * np.eye(xtx.shape[0])
    try:
        weights = np.linalg.solve(xtx + ridge, x.T @ y)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(xtx + ridge) @ x.T @ y
    return np.asarray(weights, dtype=np.float64)


def _candidate_from_row(
    row: dict[str, Any],
    feature_names: tuple[str, ...],
    target_calmar: float,
) -> MLSearchCandidate:
    spec = row["spec"]
    feature_set = tuple(feature_names[int(i)] for i in spec["columns"])
    train_metrics = MLSearchMetrics(**row["train_metrics"])
    train_subperiod_metrics = tuple(
        MLSearchMetrics(**raw) for raw in row.get("train_subperiod_metrics", [])
    )
    validation_subperiod_metrics = tuple(
        MLSearchMetrics(**raw) for raw in row.get("validation_subperiod_metrics", [])
    )
    validation_metrics = (
        None if row["validation_metrics"] is None else MLSearchMetrics(**row["validation_metrics"])
    )
    return MLSearchCandidate(
        candidate_id=str(spec["candidate_id"]),
        route=str(spec["route"]),
        model=str(spec["model"]),
        feature_set=feature_set,
        threshold=float(spec["threshold"]),
        direction=int(spec["direction"]),
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        rule=(
            f"{spec['model']} horizon {int(spec.get('horizon', 1))}d "
            f"{spec.get('target_type', 'direction')} over {len(feature_set)} features; "
            f"long if score * direction >= {float(spec['threshold']):.6f}, else short"
        ),
        train_subperiod_metrics=train_subperiod_metrics,
        validation_subperiod_metrics=validation_subperiod_metrics,
        positions_signature=str(row.get("positions_signature", "")),
        behavior_vector=tuple(float(value) for value in row.get("behavior_vector", ())),
        horizon=int(spec.get("horizon", 1)),
        target_type=str(spec.get("target_type", "direction")),
        smoothing=int(spec.get("smoothing", 1)),
        model_params=dict(spec.get("params", {})),
        seed=None if spec.get("seed") is None else int(spec.get("seed")),
        robustness=dict(row.get("robustness", {})),
        feature_importance={
            feature_set[int(idx)]: float(value)
            for idx, value in row.get("feature_importance", {}).items()
            if int(idx) < len(feature_set)
        },
    )


def _candidate_specs(
    rng: Any,
    feature_groups: dict[str, list[str]],
    max_candidates: int,
    *,
    include_classic_ml: bool,
    models: tuple[str, ...] | list[str] | None = None,
    target_calmar: float,
    validation_target_calmar: float | None = 1.0,
    min_train_subperiod_calmar: float,
    min_validation_subperiod_calmar: float | None = None,
    min_train_cagr: float | None = None,
    min_validation_cagr: float | None = None,
    max_train_mdd: float | None = None,
    max_validation_mdd: float | None = None,
    max_train_calmar: float | None = None,
    min_train_annual_return: float | None = None,
    min_validation_annual_return: float | None = None,
    min_train_annual_calmar: float | None = None,
    min_validation_annual_calmar: float | None = None,
    max_train_validation_calmar_ratio: float | None = None,
    min_validation_excess_pvalue: float | None = None,
    min_validation_bootstrap_calmar_p05: float | None = None,
    min_validation_bootstrap_excess_calmar_p05: float | None = None,
    max_validation_random_baseline_pvalue: float | None = None,
    min_validation_deflated_sharpe: float | None = None,
    max_validation_pbo: float | None = None,
    min_feature_ablation_validation_calmar: float | None = None,
    min_validation_regime_calmar: float | None = None,
    max_validation_trade_concentration: float | None = None,
    statistical_bootstrap_paths: int = 300,
    statistical_bootstrap_block: int = 21,
    statistical_random_shuffles: int = 300,
    statistical_pbo_splits: int = 8,
    min_trades_per_year: float = 0.5,
    max_trades_per_year: float | None = None,
    min_long_fraction: float | None = None,
    max_long_fraction: float | None = None,
    always_long_threshold: float = 0.98,
    always_short_threshold: float = 0.02,
    max_features_per_candidate: int | None = None,
    complexity_penalty: float = 0.0,
    anti_overfit_patterns: tuple[dict[str, Any], ...] = tuple(),
    early_stability_screen: bool = False,
    simple_survivors_mode: bool = False,
    family_weights: dict[str, float] | None = None,
    candidate_start: int = 0,
    defer_robustness_until_basic_pass: bool = False,
    effective_dsr_trials: int | None = None,
) -> list[dict[str, Any]]:
    all_features = feature_groups["all"]
    name_to_idx = {name: idx for idx, name in enumerate(all_features)}
    group_names = [name for name in feature_groups if name != "all" and feature_groups[name]]
    selected_models = list(_parse_models(tuple(models or ("corr", "ridge", "logistic", "forest"))))
    if not include_classic_ml:
        selected_models = [model for model in selected_models if model in {"corr", "ridge"}]
    if simple_survivors_mode:
        selected_models = [model for model in selected_models if model in {"corr", "ridge", "logistic"}]
        if not selected_models:
            selected_models = ["logistic"]
        max_features_per_candidate = min(4, int(max_features_per_candidate or 4))
    thresholds = (0.0, 0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
    horizons = (1, 2, 5, 10, 21)
    target_types = ("direction", "strong_return", "avoid_drawdown")
    smoothings = (1, 3, 5)
    specs: list[dict[str, Any]] = []
    for i in range(max_candidates):
        if group_names and i < len(group_names):
            pool_name = group_names[i]
            selected_names = feature_groups[pool_name]
        else:
            pool_name = (
                _rng_weighted_choice(rng, group_names, family_weights)
                if group_names
                else "all"
            )
            pool = feature_groups.get(pool_name, all_features) or all_features
            upper = min(max(3, len(pool)), 18)
            if max_features_per_candidate is not None:
                upper = min(upper, max(1, int(max_features_per_candidate)))
            size = _rng_int(rng, 3, max(3, upper))
            selected_names = sorted(_rng_sample(rng, pool, min(size, len(pool))))
        if max_features_per_candidate is not None and len(selected_names) > int(max_features_per_candidate):
            selected_names = sorted(
                _rng_sample(rng, list(selected_names), int(max_features_per_candidate))
            )
        columns = sorted({name_to_idx[name] for name in selected_names if name in name_to_idx})
        if not columns:
            columns = list(range(len(all_features)))
        model = str(_rng_choice(rng, selected_models))
        params = _model_params(model, rng)
        specs.append(
            {
                "candidate_id": f"ml-{int(candidate_start) + i + 1:06d}",
                "route": "classic_ml",
                "model": model,
                "feature_pool": pool_name,
                "columns": columns,
                "threshold": float(_rng_choice(rng, thresholds)),
                "direction": int(_rng_choice(rng, (-1, 1))),
                "horizon": int(_rng_choice(rng, horizons)),
                "target_type": str(_rng_choice(rng, target_types)),
                "smoothing": int(_rng_choice(rng, smoothings)),
                "params": params,
                "alpha": 10 ** _rng_uniform(rng, -3, 2),
                "c": 10 ** _rng_uniform(rng, -2, 1),
                "n_estimators": int(params.get("n_estimators", _rng_choice(rng, (40, 80, 120)))),
                "max_depth": int(params.get("max_depth", _rng_choice(rng, (2, 3, 4)))),
                "min_samples_leaf": int(_rng_choice(rng, (10, 20, 40))),
                "seed": _rng_int(rng, 1, 2_000_000_000),
                "target_calmar": float(target_calmar),
                "validation_target_calmar": None
                if validation_target_calmar is None
                else float(validation_target_calmar),
                "min_train_subperiod_calmar": float(min_train_subperiod_calmar),
                "min_validation_subperiod_calmar": None
                if min_validation_subperiod_calmar is None
                else float(min_validation_subperiod_calmar),
                "min_train_cagr": None
                if min_train_cagr is None
                else float(min_train_cagr),
                "min_validation_cagr": None
                if min_validation_cagr is None
                else float(min_validation_cagr),
                "max_train_mdd": None
                if max_train_mdd is None
                else float(max_train_mdd),
                "max_validation_mdd": None
                if max_validation_mdd is None
                else float(max_validation_mdd),
                "max_train_calmar": None
                if max_train_calmar is None
                else float(max_train_calmar),
                "min_train_annual_return": None
                if min_train_annual_return is None
                else float(min_train_annual_return),
                "min_validation_annual_return": None
                if min_validation_annual_return is None
                else float(min_validation_annual_return),
                "min_train_annual_calmar": None
                if min_train_annual_calmar is None
                else float(min_train_annual_calmar),
                "min_validation_annual_calmar": None
                if min_validation_annual_calmar is None
                else float(min_validation_annual_calmar),
                "max_train_validation_calmar_ratio": None
                if max_train_validation_calmar_ratio is None
                else float(max_train_validation_calmar_ratio),
                "min_validation_excess_pvalue": None
                if min_validation_excess_pvalue is None
                else float(min_validation_excess_pvalue),
                "min_validation_bootstrap_calmar_p05": None
                if min_validation_bootstrap_calmar_p05 is None
                else float(min_validation_bootstrap_calmar_p05),
                "min_validation_bootstrap_excess_calmar_p05": None
                if min_validation_bootstrap_excess_calmar_p05 is None
                else float(min_validation_bootstrap_excess_calmar_p05),
                "max_validation_random_baseline_pvalue": None
                if max_validation_random_baseline_pvalue is None
                else float(max_validation_random_baseline_pvalue),
                "min_validation_deflated_sharpe": None
                if min_validation_deflated_sharpe is None
                else float(min_validation_deflated_sharpe),
                "max_validation_pbo": None
                if max_validation_pbo is None
                else float(max_validation_pbo),
                "min_feature_ablation_validation_calmar": None
                if min_feature_ablation_validation_calmar is None
                else float(min_feature_ablation_validation_calmar),
                "min_validation_regime_calmar": None
                if min_validation_regime_calmar is None
                else float(min_validation_regime_calmar),
                "max_validation_trade_concentration": None
                if max_validation_trade_concentration is None
                else float(max_validation_trade_concentration),
                "statistical_bootstrap_paths": int(statistical_bootstrap_paths),
                "statistical_bootstrap_block": int(statistical_bootstrap_block),
                "statistical_random_shuffles": int(statistical_random_shuffles),
                "statistical_pbo_splits": int(statistical_pbo_splits),
                "defer_robustness_until_basic_pass": bool(defer_robustness_until_basic_pass),
                "effective_dsr_trials": None
                if effective_dsr_trials is None
                else int(effective_dsr_trials),
                "complexity_penalty": float(complexity_penalty),
                "early_stability_screen": bool(early_stability_screen),
                "simple_survivors_mode": bool(simple_survivors_mode),
                "repeated_failure_penalty": _anti_overfit_penalty(
                    tuple(selected_names),
                    model=model,
                    family=pool_name,
                    patterns=anti_overfit_patterns,
                ),
                "n_trials": int(max_candidates),
                "min_trades_per_year": float(min_trades_per_year),
                "max_trades_per_year": None
                if max_trades_per_year is None
                else float(max_trades_per_year),
                "min_long_fraction": None
                if min_long_fraction is None
                else float(min_long_fraction),
                "max_long_fraction": None
                if max_long_fraction is None
                else float(max_long_fraction),
                "always_long_threshold": float(always_long_threshold),
                "always_short_threshold": float(always_short_threshold),
            }
        )
    return specs


def _parse_models(models: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
    raw = (models.split(",") if isinstance(models, str) else list(models))
    aliases = {"lgbm": "lightgbm", "xgb": "xgboost", "logreg": "logistic"}
    parsed: list[str] = []
    for item in raw:
        model = aliases.get(str(item).strip().lower(), str(item).strip().lower())
        if model not in {"lightgbm", "xgboost", "logistic", "forest", "ridge", "corr"}:
            raise ValueError(f"unknown ml-search model: {item}")
        if model not in parsed:
            parsed.append(model)
    if not parsed:
        raise ValueError("at least one ml-search model is required")
    return tuple(parsed)


def _model_params(model: str, rng: Any) -> dict[str, Any]:
    if model == "lightgbm":
        return dict(
            _rng_choice(
                rng,
                (
                    {"n_estimators": 120, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 3},
                    {"n_estimators": 240, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 5},
                    {"n_estimators": 320, "learning_rate": 0.02, "num_leaves": 31, "max_depth": -1},
                    {
                        "n_estimators": 160,
                        "learning_rate": 0.03,
                        "num_leaves": 7,
                        "max_depth": 2,
                        "min_child_samples": 200,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                    },
                    {
                        "n_estimators": 420,
                        "learning_rate": 0.01,
                        "num_leaves": 31,
                        "max_depth": 5,
                        "min_child_samples": 80,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "reg_alpha": 0.1,
                        "reg_lambda": 2.0,
                    },
                ),
            )
        )
    if model == "xgboost":
        return dict(
            _rng_choice(
                rng,
                (
                    {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8},
                    {
                        "n_estimators": 240,
                        "learning_rate": 0.03,
                        "max_depth": 4,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                    },
                    {
                        "n_estimators": 320,
                        "learning_rate": 0.02,
                        "max_depth": 5,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                    },
                    {
                        "n_estimators": 160,
                        "learning_rate": 0.03,
                        "max_depth": 2,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "min_child_weight": 20,
                    },
                    {
                        "n_estimators": 420,
                        "learning_rate": 0.01,
                        "max_depth": 5,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "min_child_weight": 5,
                        "reg_alpha": 0.1,
                        "reg_lambda": 2.0,
                    },
                ),
            )
        )
    return {}


def _rng_choice(rng: Any, values: tuple[Any, ...] | list[Any]) -> Any:
    values = list(values)
    if hasattr(rng, "choice"):
        return values[int(rng.choice(len(values)))] if not isinstance(rng, random.Random) else rng.choice(values)
    return random.choice(values)


def _rng_weighted_choice(
    rng: Any,
    values: tuple[str, ...] | list[str],
    weights: dict[str, float] | None,
) -> str:
    values = list(values)
    if not values:
        return "all"
    if not weights:
        return str(_rng_choice(rng, values))
    raw = np.asarray([max(0.0, float(weights.get(value, 0.0))) for value in values])
    if not np.isfinite(raw).all() or float(raw.sum()) <= 0.0:
        return str(_rng_choice(rng, values))
    probs = raw / float(raw.sum())
    if isinstance(rng, random.Random):
        return str(rng.choices(values, weights=raw.tolist(), k=1)[0])
    if hasattr(rng, "choice"):
        return str(values[int(rng.choice(len(values), p=probs))])
    return str(random.choices(values, weights=raw.tolist(), k=1)[0])


def _rng_sample(rng: Any, values: list[Any], size: int) -> list[Any]:
    if isinstance(rng, random.Random):
        return rng.sample(values, size)
    if hasattr(rng, "choice"):
        idx = rng.choice(len(values), size=size, replace=False)
        return [values[int(i)] for i in idx]
    return random.sample(values, size)


def _rng_int(rng: Any, low: int, high: int) -> int:
    if isinstance(rng, random.Random):
        return rng.randint(low, high)
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high + 1))
    return random.randint(low, high)


def _rng_uniform(rng: Any, low: float, high: float) -> float:
    if isinstance(rng, random.Random):
        return rng.uniform(low, high)
    if hasattr(rng, "uniform"):
        return float(rng.uniform(low, high))
    return random.uniform(low, high)


def _adaptive_family_reward(row: dict[str, Any]) -> float:
    reason = row.get("rejection_reason")
    if reason is None and row.get("validation_metrics") is not None:
        return 20.0
    if row.get("validation_metrics") is not None:
        text = str(reason or "")
        if text.startswith("validation_"):
            return 6.0
        return 4.0
    text = str(reason or "")
    if text in {"train_validation_calmar_ratio", "train_subperiod_calmar"}:
        return 2.0
    if text in {"train_total_metrics", "train_annual_return", "train_annual_calmar"}:
        return 1.0
    train_metrics = row.get("train_metrics") or {}
    spec = row.get("spec") or {}
    try:
        train_calmar = float(train_metrics.get("calmar", 0.0))
        target = max(1e-9, float(spec.get("target_calmar", 1.0)))
    except Exception:
        return 0.0
    if text == "train_calmar" and train_calmar > 0.0:
        return min(0.9, train_calmar / target)
    return 0.0


def _adaptive_family_weights(
    group_names: list[str],
    attempts: Counter[str],
    rewards: Counter[str],
    *,
    min_weight: float,
    reward_scale: float,
    penalties: dict[str, float] | None = None,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for name in group_names:
        tries = max(1, int(attempts.get(name, 0)))
        score = float(rewards.get(name, 0.0)) / tries
        penalty = 1.0
        if penalties and name in penalties:
            penalty = max(0.01, float(penalties[name]))
        weights[name] = (
            max(0.01, float(min_weight))
            + max(0.0, score) * max(0.01, float(reward_scale))
        ) * penalty
    return weights


def _feature_groups(feature_names: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "all": list(feature_names),
        "returns": [name for name in feature_names if "ret" in name or "lag" in name],
        "rolling": [name for name in feature_names if name.startswith("roll_")],
        "technicals": [name for name in feature_names if name.startswith(("rsi", "macd", "bb_"))],
        "volatility": [name for name in feature_names if "vol" in name or "std" in name],
        "price_action": [name for name in feature_names if name.startswith("pa_")],
        "microstructure": [name for name in feature_names if name in {"cs_spread", "signed_volume"}],
        "pending": [name for name in feature_names if name.startswith("pending_")],
        "pending_market": [
            name
            for name in feature_names
            if name.startswith("pending_")
            and any(
                token in name
                for token in (
                    "vix",
                    "vvix",
                    "skew",
                    "qqq",
                    "iwm",
                    "rsp",
                    "efa",
                    "eem",
                    "gld",
                    "uso",
                    "hyg",
                    "lqd",
                    "xle",
                    "xlf",
                    "xlk",
                    "xlu",
                    "xlp",
                    "xly",
                )
            )
        ],
        "pending_calendar": [
            name for name in feature_names if name.startswith("pending_calendar_")
        ],
        "pending_price_structure": [
            name
            for name in feature_names
            if name.startswith("pending_")
            and any(
                token in name
                for token in (
                    "range",
                    "gap",
                    "overnight",
                    "intraday",
                    "atr",
                    "parkinson",
                    "garman",
                    "rogers",
                )
            )
        ],
    }
    return groups


def _literature_feature_groups(
    feature_names: list[str],
    literature_ideas: tuple[dict[str, Any], ...],
) -> dict[str, list[str]]:
    if not literature_ideas:
        return {}
    groups: dict[str, list[str]] = {}
    for index, idea in enumerate(literature_ideas, start=1):
        if not isinstance(idea, dict):
            continue
        text = " ".join(
            (
                str(idea.get("idea_id", "")),
                str(idea.get("rule_family", "")),
                str(idea.get("hypothesis", "")),
                " ".join(str(item) for item in idea.get("features", ()) or ()),
            )
        ).lower()
        selected = _features_for_literature_text(feature_names, text)
        if selected:
            safe_id = "".join(
                ch if ch.isalnum() else "_"
                for ch in str(idea.get("idea_id", f"idea_{index}")).lower()
            ).strip("_")[:40]
            groups[f"literature_{index:02d}_{safe_id or 'idea'}"] = selected
    if groups:
        all_literature = sorted({name for values in groups.values() for name in values})
        groups["literature_all"] = all_literature
    return groups


def _features_for_literature_text(feature_names: list[str], text: str) -> list[str]:
    aliases = {
        ("vix", "volatility", "variance", "risk", "drawdown", "crash", "tail"): (
            "vix",
            "vvix",
            "skew",
            "vol",
            "std",
            "atr",
            "parkinson",
            "garman",
            "rogers",
            "drawdown",
            "dd",
            "tail",
        ),
        ("credit", "spread", "hyg", "lqd", "stress", "financial condition"): (
            "credit",
            "spread",
            "hyg",
            "lqd",
            "stress",
            "nfci",
            "financial",
        ),
        ("momentum", "trend", "moving average", "time series"): (
            "ret",
            "lag",
            "roll",
            "macd",
            "rsi",
            "bb_",
            "trend",
            "momentum",
        ),
        ("yield", "rate", "treasury", "curve", "term structure", "term spread"): (
            "yield",
            "rate",
            "treasury",
            "curve",
            "term",
            "tnx",
            "fred",
        ),
        ("sector", "defensive", "cyclical", "industry"): (
            "sector",
            "xlu",
            "xlp",
            "xly",
            "xlf",
            "xlk",
            "xle",
            "xlv",
            "defensive",
            "cyclical",
        ),
        ("breadth", "advance decline", "equal weight", "small cap"): (
            "breadth",
            "advance",
            "decline",
            "rsp",
            "iwm",
            "equal",
            "small",
        ),
        ("macro", "inflation", "unemployment", "business cycle", "industrial production"): (
            "macro",
            "inflation",
            "cpi",
            "unemployment",
            "claims",
            "industrial",
            "pmi",
            "fred",
        ),
        ("liquidity", "funding", "market liquidity"): (
            "liquidity",
            "funding",
            "volume",
            "spread",
        ),
    }
    wanted: set[str] = set()
    for triggers, tokens in aliases.items():
        if any(trigger in text for trigger in triggers):
            wanted.update(tokens)
    if not wanted:
        wanted.update(("ret", "roll", "pa_", "pending_"))
    selected = [
        name
        for name in feature_names
        if any(token in name.lower() for token in wanted)
    ]
    return selected[:80]


def _split_frame(frame: pd.DataFrame, config: MLSearchConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.loc[: config.train_end]
    validation = frame.loc[config.validation_start: config.validation_end]
    if train.empty:
        raise ValueError("train period is empty")
    if validation.empty:
        raise ValueError("validation period is empty")
    return train, validation


def _next_returns(close: pd.Series) -> pd.Series:
    return close.shift(-1) / close - 1.0


def _aligned_xy(
    features: pd.DataFrame,
    next_returns: pd.Series,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, pd.DatetimeIndex]:
    joined = features.join(next_returns.rename("_next_return")).replace([np.inf, -np.inf], np.nan)
    joined = joined.dropna(subset=["_next_return"])
    feature_cols = tuple(column for column in joined.columns if column != "_next_return")
    x = joined[list(feature_cols)].to_numpy(dtype=np.float64)
    y = joined["_next_return"].to_numpy(dtype=np.float64)
    valid_rows = np.isfinite(y)
    return x[valid_rows], y[valid_rows], feature_cols, y[valid_rows], joined.index[valid_rows]


def _aligned_validation(
    features: pd.DataFrame,
    next_returns: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    joined = features.join(next_returns.rename("_next_return")).replace([np.inf, -np.inf], np.nan)
    joined = joined.dropna(subset=["_next_return"])
    return joined.drop(columns=["_next_return"]).to_numpy(dtype=np.float64), joined["_next_return"].to_numpy(dtype=np.float64)


def _aligned_index(features: pd.DataFrame, next_returns: pd.Series) -> pd.DatetimeIndex:
    joined = features.join(next_returns.rename("_next_return")).replace([np.inf, -np.inf], np.nan)
    joined = joined.dropna(subset=["_next_return"])
    return pd.DatetimeIndex(joined.index)


def _subperiod_masks(index: pd.DatetimeIndex, count: int) -> tuple[np.ndarray, ...]:
    if count <= 1:
        return (np.ones(len(index), dtype=bool),)
    positions = np.array_split(np.arange(len(index)), int(count))
    masks: list[np.ndarray] = []
    for pos in positions:
        mask = np.zeros(len(index), dtype=bool)
        mask[pos] = True
        masks.append(mask)
    return tuple(masks)


def _annual_return_masks(index: pd.DatetimeIndex) -> tuple[np.ndarray, ...]:
    years = pd.Index(index).year
    masks: list[np.ndarray] = []
    for year in sorted(set(int(value) for value in years)):
        masks.append(np.asarray(years == year, dtype=bool))
    return tuple(masks)


def _train_annual_return_ok(
    next_returns: np.ndarray,
    positions: np.ndarray,
    *,
    min_return: float | None,
) -> bool:
    return _period_annual_return_ok(
        next_returns,
        positions,
        _TRAIN_ANNUAL_RETURN_MASKS,
        min_return=min_return,
    )


def _period_annual_return_ok(
    next_returns: np.ndarray,
    positions: np.ndarray,
    masks: tuple[np.ndarray, ...],
    *,
    min_return: float | None,
) -> bool:
    if min_return is None:
        return True
    returns = np.asarray(next_returns, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    for mask in masks:
        if not bool(np.any(mask)):
            continue
        annual_return = float(np.prod(1.0 + pos[mask] * returns[mask]) - 1.0)
        if annual_return < float(min_return):
            return False
    return True


def _period_annual_calmar_ok(
    next_returns: np.ndarray,
    positions: np.ndarray,
    masks: tuple[np.ndarray, ...],
    *,
    min_calmar: float | None,
) -> bool:
    if min_calmar is None:
        return True
    returns = np.asarray(next_returns, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    for mask in masks:
        if not bool(np.any(mask)):
            continue
        metrics = metrics_from_next_returns(returns[mask], pos[mask])
        if metrics.calmar < float(min_calmar):
            return False
    return True


def _train_validation_ratio_ok(
    train_metrics: MLSearchMetrics,
    validation_metrics: MLSearchMetrics,
    *,
    max_ratio: float | None,
) -> bool:
    if max_ratio is None:
        return True
    valid_calmar = max(float(validation_metrics.calmar), 1e-9)
    if train_metrics.calmar <= 0.0:
        return True
    return float(train_metrics.calmar) / valid_calmar <= float(max_ratio)


def _validation_statistical_ok(
    robustness: dict[str, Any],
    *,
    max_excess_pvalue: float | None,
    min_bootstrap_calmar_p05: float | None,
    min_bootstrap_excess_calmar_p05: float | None,
    max_random_baseline_pvalue: float | None,
    min_deflated_sharpe: float | None,
    max_pbo: float | None,
    min_feature_ablation_validation_calmar: float | None,
    min_regime_calmar: float | None,
    max_trade_concentration: float | None,
) -> str | None:
    if (
        max_excess_pvalue is not None
        and float(robustness.get("excess_pvalue", 1.0)) > float(max_excess_pvalue)
    ):
        return "validation_excess_pvalue"
    if (
        min_bootstrap_calmar_p05 is not None
        and float(robustness.get("bootstrap_calmar_p05", -math.inf))
        < float(min_bootstrap_calmar_p05)
    ):
        return "validation_bootstrap_calmar_p05"
    if (
        min_bootstrap_excess_calmar_p05 is not None
        and float(robustness.get("bootstrap_excess_calmar_p05", -math.inf))
        < float(min_bootstrap_excess_calmar_p05)
    ):
        return "validation_bootstrap_excess_calmar_p05"
    if (
        max_random_baseline_pvalue is not None
        and float(robustness.get("random_baseline_pvalue", 1.0))
        > float(max_random_baseline_pvalue)
    ):
        return "validation_random_baseline_pvalue"
    if (
        min_deflated_sharpe is not None
        and float(robustness.get("deflated_sharpe", -math.inf))
        < float(min_deflated_sharpe)
    ):
        return "validation_deflated_sharpe"
    if max_pbo is not None and float(robustness.get("pbo", 1.0)) > float(max_pbo):
        return "validation_pbo"
    if (
        min_feature_ablation_validation_calmar is not None
        and float(robustness.get("feature_ablation_validation_calmar", -math.inf))
        < float(min_feature_ablation_validation_calmar)
    ):
        return "validation_feature_ablation"
    if (
        min_regime_calmar is not None
        and float(robustness.get("regime_min_calmar", -math.inf)) < float(min_regime_calmar)
    ):
        return "validation_regime_calmar"
    if (
        max_trade_concentration is not None
        and float(robustness.get("trade_concentration_top5", 1.0))
        > float(max_trade_concentration)
    ):
        return "validation_trade_concentration"
    return None


def _validation_robustness_summary(
    next_returns: np.ndarray,
    positions: np.ndarray,
    *,
    paths: int,
    block: int,
    seed: int,
    n_trials: int,
    random_shuffles: int,
    pbo_splits: int,
) -> dict[str, Any]:
    returns = np.asarray(next_returns, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    strategy = pos * returns
    benchmark = returns
    excess = strategy - benchmark
    paths = max(10, int(paths))
    block = max(1, int(block))
    rng = np.random.default_rng(int(seed))
    calmar_values = np.empty(paths, dtype=np.float64)
    excess_calmar_values = np.empty(paths, dtype=np.float64)
    for idx in range(paths):
        sample_idx = _circular_block_indices(len(strategy), block, rng)
        calmar_values[idx] = _calmar_from_returns(strategy[sample_idx])
        excess_calmar_values[idx] = _calmar_from_returns(excess[sample_idx])
    strategy_metrics = compute_metrics(strategy, ppy=252)
    dsr = deflated_sharpe_annualized(
        float(strategy_metrics.sharpe),
        max(2, int(n_trials)),
        max(2, int(strategy_metrics.n_periods)),
        252,
        skew=float(strategy_metrics.skew),
        kurtosis=float(strategy_metrics.kurtosis),
        min_dsr=0.0,
    )
    random_pvalue = _random_baseline_calmar_pvalue(
        returns,
        pos,
        shuffles=max(10, int(random_shuffles)),
        seed=seed + 101,
    )
    pbo = _pbo_against_random_baselines(
        returns,
        pos,
        shuffles=max(10, int(random_shuffles)),
        splits=max(4, int(pbo_splits)),
        seed=seed + 202,
    )
    regime = _regime_calmar_summary(returns, pos)
    concentration = _trade_concentration_top5(returns, pos)
    return {
        "excess_pvalue": _one_sided_mean_pvalue(excess),
        "bootstrap_calmar_p05": float(np.quantile(calmar_values, 0.05)),
        "bootstrap_calmar_p50": float(np.quantile(calmar_values, 0.50)),
        "bootstrap_excess_calmar_p05": float(np.quantile(excess_calmar_values, 0.05)),
        "bootstrap_excess_calmar_p50": float(np.quantile(excess_calmar_values, 0.50)),
        "random_baseline_pvalue": random_pvalue,
        "deflated_sharpe": float(dsr.dsr),
        "pbo": pbo,
        "regime_min_calmar": float(min(regime.values())) if regime else 0.0,
        "regime_calmar": regime,
        "trade_concentration_top5": concentration,
        "bootstrap_paths": int(paths),
        "bootstrap_block": int(block),
    }


def _circular_block_indices(length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        return np.asarray([], dtype=int)
    pieces: list[np.ndarray] = []
    while sum(len(piece) for piece in pieces) < length:
        start = int(rng.integers(0, length))
        pieces.append((np.arange(start, start + block) % length).astype(int))
    return np.concatenate(pieces)[:length]


def _random_baseline_calmar_pvalue(
    returns: np.ndarray,
    positions: np.ndarray,
    *,
    shuffles: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    candidate = _calmar_from_returns(pos * returns)
    samples = np.empty(max(1, int(shuffles)), dtype=np.float64)
    for idx in range(len(samples)):
        shuffled = pos.copy()
        rng.shuffle(shuffled)
        samples[idx] = _calmar_from_returns(shuffled * returns)
    return float((np.sum(samples >= candidate) + 1.0) / (len(samples) + 1.0))


def _pbo_against_random_baselines(
    returns: np.ndarray,
    positions: np.ndarray,
    *,
    shuffles: int,
    splits: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    if len(returns) < 20:
        return 1.0
    split_count = max(4, int(splits))
    if split_count % 2:
        split_count += 1
    blocks = [block for block in np.array_split(np.arange(len(returns)), split_count) if len(block)]
    if len(blocks) < 4:
        return 1.0
    random_positions = []
    for _ in range(max(2, int(shuffles))):
        shuffled = pos.copy()
        rng.shuffle(shuffled)
        random_positions.append(shuffled)
    matrix = np.column_stack([pos * returns, *[rp * returns for rp in random_positions]])
    bad = 0
    total = 0
    half = len(blocks) // 2
    # Deterministic balanced sample: each adjacent half is IS once.
    for start in range(len(blocks)):
        is_blocks = [blocks[(start + offset) % len(blocks)] for offset in range(half)]
        oos_blocks = [blocks[(start + half + offset) % len(blocks)] for offset in range(half)]
        is_idx = np.concatenate(is_blocks)
        oos_idx = np.concatenate(oos_blocks)
        is_scores = np.asarray([_calmar_from_returns(matrix[is_idx, col]) for col in range(matrix.shape[1])])
        oos_scores = np.asarray([_calmar_from_returns(matrix[oos_idx, col]) for col in range(matrix.shape[1])])
        chosen = int(np.nanargmax(is_scores))
        rank = float(np.mean(oos_scores <= oos_scores[chosen]))
        if rank <= 0.5:
            bad += 1
        total += 1
    return float(bad / max(total, 1))


def _regime_calmar_summary(returns: np.ndarray, positions: np.ndarray) -> dict[str, float]:
    returns = np.asarray(returns, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    finite = returns[np.isfinite(returns)]
    if len(finite) < 20:
        return {}
    low = float(np.quantile(finite, 0.10))
    abs_median = float(np.quantile(np.abs(finite), 0.50))
    masks = {
        "up": returns > 0.0,
        "down": returns < 0.0,
        "crisis": returns <= low,
        "quiet": np.abs(returns) <= abs_median,
    }
    out: dict[str, float] = {}
    for name, mask in masks.items():
        if int(np.sum(mask)) < 5:
            out[name] = 0.0
        else:
            out[name] = _calmar_from_returns(pos[mask] * returns[mask])
    return out


def _trade_concentration_top5(returns: np.ndarray, positions: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    if len(returns) == 0:
        return 1.0
    change_points = np.flatnonzero(np.abs(np.diff(pos, prepend=pos[0])) > 0.0)
    starts = np.r_[0, change_points]
    ends = np.r_[change_points, len(returns)]
    trade_returns: list[float] = []
    for start, end in zip(starts, ends):
        if end <= start:
            continue
        trade_returns.append(float(np.prod(1.0 + pos[start:end] * returns[start:end]) - 1.0))
    positive = sorted((value for value in trade_returns if value > 0.0), reverse=True)
    total_positive = float(sum(positive))
    if total_positive <= 1e-12:
        return 1.0
    return float(sum(positive[:5]) / total_positive)


def _feature_ablation_calmar(
    spec: dict[str, Any],
    cols: np.ndarray,
    importance: dict[str, float],
    *,
    validation_target_shape: int,
) -> float:
    if _X_TRAIN is None or _Y_TRAIN is None or _RET_VALID_NEXT is None or _X_VALID is None:
        return -math.inf
    if len(cols) <= 1:
        return -math.inf
    if importance:
        remove_local = max(importance, key=lambda key: abs(float(importance[key])))
        remove_idx = int(remove_local)
    else:
        remove_idx = 0
    kept = np.asarray([int(col) for idx, col in enumerate(cols) if idx != remove_idx], dtype=int)
    if len(kept) == 0:
        return -math.inf
    ablated = dict(spec)
    ablated["columns"] = kept.tolist()
    x_train = _fill_matrix(_X_TRAIN[:, kept])
    x_valid = _fill_matrix(_X_VALID[:, kept])
    payload = _fit_predict_payload(ablated, x_train, _Y_TRAIN, x_train, x_valid)
    scores = _smooth_scores(
        payload["validation_scores"],
        smoothing=int(spec.get("smoothing", 1)),
    )
    if len(scores) != int(validation_target_shape):
        return -math.inf
    positions = positions_from_scores(
        scores,
        threshold=float(spec["threshold"]),
        direction=int(spec["direction"]),
    )
    return float(metrics_from_next_returns(_RET_VALID_NEXT, positions).calmar)


def _one_sided_mean_pvalue(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        return 1.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std <= 1e-12:
        return 0.0 if mean > 0.0 else 1.0
    z = mean / (std / math.sqrt(float(len(arr))))
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def _calmar_from_returns(returns: np.ndarray) -> float:
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    equity = np.cumprod(1.0 + arr)
    final = float(equity[-1])
    years = _years_from_rows(len(arr))
    cagr = final ** (1.0 / years) - 1.0 if final > 0.0 else -1.0
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    mdd = float(np.min(drawdown)) if len(drawdown) else 0.0
    return float(cagr / abs(mdd)) if abs(mdd) > 1e-12 else (999.0 if cagr > 0 else 0.0)


def _metrics_total_ok(
    metrics: MLSearchMetrics,
    *,
    min_cagr: float | None,
    max_mdd: float | None,
) -> bool:
    if min_cagr is not None and metrics.cagr < float(min_cagr):
        return False
    if max_mdd is not None and metrics.mdd < -abs(float(max_mdd)):
        return False
    return True


def _train_calmar_ceiling_reason(
    metrics: MLSearchMetrics,
    *,
    max_train_calmar: float | None,
) -> str | None:
    if max_train_calmar is None:
        return None
    if float(metrics.calmar) > float(max_train_calmar):
        return "train_calmar_too_high"
    return None


def _fill_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def _years_from_rows(rows: int) -> float:
    return max(float(rows) / 252.0, 1e-9)


def _run_kronos_challenger(config: MLSearchConfig) -> list[str]:
    try:
        from aurora.research.kronos_tool import KronosToolConfig, run_kronos_search

        kronos_report = run_kronos_search(
            KronosToolConfig(
                run_id=f"{config.run_id}-kronos",
                symbol=config.symbol,
                library=config.library,
                target_calmar=config.target_calmar,
                validation_target_calmar=config.validation_target_calmar,
                run_root=config.run_root,
                allow_volume=True,
                train_only=True,
                no_costs=True,
                max_windows=200,
            )
        )
        if kronos_report.objective_met:
            return ["kronos objective met in its own artifact folder; inspect kronos report"]
        return []
    except Exception as exc:
        return [f"kronos skipped: {exc}"]


def _output_dir(config: MLSearchConfig) -> Path:
    root = Path(config.run_root) if config.run_root else base_data_dir() / "agent_loop"
    return root / config.run_id / "ml_search"


def _status_payload(
    config: MLSearchConfig,
    output_dir: Path,
    status: str,
    *,
    candidates_evaluated: int = 0,
    batches_completed: int = 0,
    best_train: MLSearchCandidate | None = None,
    best_validation: MLSearchCandidate | None = None,
    objective_candidates: tuple[MLSearchCandidate, ...] = tuple(),
    objective_met: bool = False,
    used_columns: tuple[str, ...] = tuple(),
    route_errors: tuple[str, ...] = tuple(),
    behavior_counts: dict[str, int] | None = None,
    rejection_counts: dict[str, int] | None = None,
    adaptive_family_attempts: dict[str, int] | None = None,
    adaptive_family_rewards: dict[str, float] | None = None,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    return {
        "status": status,
        "locked_opened": False,
        "selection_phase": "train",
        "validation_used_for_selection": False,
        "validation_examined_after_train_pass": config.validation_target_calmar is not None,
        "run_id": config.run_id,
        "symbol": config.symbol,
        "workers": config.workers,
        "target_calmar": config.target_calmar,
        "validation_target_calmar": config.validation_target_calmar,
        "target_objective_count": config.target_objective_count,
        "objective_candidates_found": len(objective_candidates),
        "objective_candidates": [candidate.to_dict() for candidate in objective_candidates],
        "min_feature_jaccard_distance": config.min_feature_jaccard_distance,
        "min_behavior_distance": config.min_behavior_distance,
        "min_trades_per_year": config.min_trades_per_year,
        "max_trades_per_year": config.max_trades_per_year,
        "min_long_fraction": config.min_long_fraction,
        "max_long_fraction": config.max_long_fraction,
        "always_long_threshold": config.always_long_threshold,
        "always_short_threshold": config.always_short_threshold,
        "train_subperiod_count": config.train_subperiod_count,
        "validation_subperiod_count": config.validation_subperiod_count
        if config.validation_subperiod_count is not None
        else config.train_subperiod_count,
        "min_train_subperiod_calmar": config.min_train_subperiod_calmar,
        "min_validation_subperiod_calmar": config.min_validation_subperiod_calmar,
        "min_train_cagr": config.min_train_cagr,
        "min_validation_cagr": config.min_validation_cagr,
        "max_train_mdd": config.max_train_mdd,
        "max_validation_mdd": config.max_validation_mdd,
        "max_train_calmar": config.max_train_calmar,
        "min_train_annual_return": config.min_train_annual_return,
        "min_validation_annual_return": config.min_validation_annual_return,
        "min_train_annual_calmar": config.min_train_annual_calmar,
        "min_validation_annual_calmar": config.min_validation_annual_calmar,
        "max_train_validation_calmar_ratio": config.max_train_validation_calmar_ratio,
        "min_validation_excess_pvalue": config.min_validation_excess_pvalue,
        "min_validation_bootstrap_calmar_p05": config.min_validation_bootstrap_calmar_p05,
        "min_validation_bootstrap_excess_calmar_p05": config.min_validation_bootstrap_excess_calmar_p05,
        "max_validation_random_baseline_pvalue": config.max_validation_random_baseline_pvalue,
        "min_validation_deflated_sharpe": config.min_validation_deflated_sharpe,
        "max_validation_pbo": config.max_validation_pbo,
        "min_feature_ablation_validation_calmar": config.min_feature_ablation_validation_calmar,
        "min_validation_regime_calmar": config.min_validation_regime_calmar,
        "max_validation_trade_concentration": config.max_validation_trade_concentration,
        "statistical_bootstrap_paths": config.statistical_bootstrap_paths,
        "statistical_bootstrap_block": config.statistical_bootstrap_block,
        "statistical_random_shuffles": config.statistical_random_shuffles,
        "statistical_pbo_splits": config.statistical_pbo_splits,
        "effective_dsr_trials": config.effective_dsr_trials,
        "time_limit_seconds": config.time_limit_seconds,
        "defer_robustness_until_basic_pass": config.defer_robustness_until_basic_pass,
        "adaptive_family_search": config.adaptive_family_search,
        "adaptive_quick_screen_candidates": config.adaptive_quick_screen_candidates,
        "adaptive_family_min_weight": config.adaptive_family_min_weight,
        "adaptive_family_reward": config.adaptive_family_reward,
        "adaptive_initial_family_attempts": dict(config.adaptive_initial_family_attempts or {}),
        "adaptive_initial_family_rewards": dict(config.adaptive_initial_family_rewards or {}),
        "penalized_feature_pools": list(config.penalized_feature_pools),
        "penalized_feature_pool_factor": config.penalized_feature_pool_factor,
        "max_features_per_candidate": config.max_features_per_candidate,
        "complexity_penalty": config.complexity_penalty,
        "anti_overfit_ledger_path": config.anti_overfit_ledger_path,
        "family_tournament_mode": config.family_tournament_mode,
        "early_stability_screen": config.early_stability_screen,
        "simple_survivors_mode": config.simple_survivors_mode,
        "models": list(config.models),
        "include_kronos": config.include_kronos,
        "include_classic_ml": config.include_classic_ml,
        "include_sequence_models": config.include_sequence_models,
        "include_pending_features": config.include_pending_features,
        "reject_same_feature_family": config.reject_same_feature_family,
        "pending_feature_library": config.pending_feature_library,
        "pending_feature_version": config.pending_feature_version,
        "literature_ideas_count": len(config.literature_ideas),
        "candidates_evaluated": candidates_evaluated,
        "batches_completed": batches_completed,
        "best_train": None if best_train is None else best_train.to_dict(),
        "best_validation": None if best_validation is None else best_validation.to_dict(),
        "objective_met": objective_met,
        "used_columns": list(used_columns),
        "route_errors": list(route_errors),
        "behavior_counts": dict(behavior_counts or {}),
        "rejection_counts": dict(rejection_counts or {}),
        "adaptive_family_attempts": dict(adaptive_family_attempts or {}),
        "adaptive_family_rewards": dict(adaptive_family_rewards or {}),
        "output_dir": str(output_dir),
        "elapsed_seconds": elapsed_seconds,
        "locked_period": (config.locked_start, "closed"),
        "updated_at_utc": _now(),
    }


def _period_tuple(frame: pd.DataFrame) -> tuple[str, str]:
    if frame.empty:
        return ("", "")
    return (str(frame.index.min().date()), str(frame.index.max().date()))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _time_limit_reached(started: float, time_limit_seconds: float | None) -> bool:
    if time_limit_seconds is None:
        return False
    return (time.perf_counter() - started) >= max(0.0, float(time_limit_seconds))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
