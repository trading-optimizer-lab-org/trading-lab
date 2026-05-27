from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_lab.annual_prediction import _build_daily_feature_frame, _political_features
from trading_lab.monthly_multi_asset import TRADABLE_ASSETS, _asset_col, _ratio_column
from trading_lab.monthly_risk import (
    LOCKED_START,
    TRAIN_END,
    VALIDATION_END,
    MonthlyRiskCandidate,
    MonthlyRiskSearchConfig,
    _candidate_exposure,
    _candidate_features,
    _json_safe,
    _normalize_daily,
    _rules_text,
)


MIN_DOWN_YEAR_RETURN = 0.05
WEEKLY_DOWN_5PCT_SCORE_MODE = "train_only_weekly_sp500_down_5pct"
WEEKLY_MAX_CALMAR_SCORE_MODE = "train_calmar_max_validation_80pct_report"
WEEKLY_MAX_SHARPE_SCORE_MODE = "train_sharpe_max_validation_80pct_report"
WEEKLY_ASSET_SELECTORS = ("momentum_4w", "momentum_13w", "momentum_26w", "momentum_52w", "low_vol_13w")
WEEKLY_METHODS = ("random_broad", "beam", "genetic", "bayesian_like", "bandit")


@dataclass(frozen=True)
class WeeklyMultiAssetCandidate:
    specs: tuple[str, ...]
    assets: tuple[str, ...]
    selector: str = "momentum_26w"
    intercept: float = 0.0
    scale: float = 1.0
    smoothing: float = 0.0


@dataclass(frozen=True)
class WeeklyMachineLearningCandidate:
    features: tuple[str, ...]
    assets: tuple[str, ...]
    selector: str = "momentum_26w"
    model: str = "ridge"
    alpha: float = 1.0
    n_estimators: int = 64
    max_depth: int = 3
    learning_rate: float = 0.05
    scale: float = 1.0
    intercept: float = 0.0
    random_seed: int = 0


def build_weekly_multi_asset_examples(
    daily: pd.DataFrame,
    *,
    benchmark_daily: pd.DataFrame | None = None,
    start_year: int = 1994,
    end_year: int | None = None,
) -> pd.DataFrame:
    data = _normalize_daily(daily)
    end_year = end_year or int(data.index.max().year)
    features = _build_daily_feature_frame(data)
    week_closes = data.resample("W-FRI").last().dropna(subset=["close"])
    down_years = _benchmark_down_years(data, benchmark_daily)
    rows: list[dict[str, object]] = []
    for index in range(len(week_closes) - 1):
        decision_date = week_closes.index[index]
        target_end = week_closes.index[index + 1]
        target_year = int(target_end.year)
        if target_year < start_year or target_year > end_year:
            continue
        decision_close = float(week_closes["close"].iloc[index])
        target_close = float(week_closes["close"].iloc[index + 1])
        if not np.isfinite(decision_close) or decision_close <= 0.0:
            continue
        feature_date = data.index[data.index <= decision_date][-1]
        feature_row = features.loc[feature_date]
        row: dict[str, object] = {
            "decision_date": decision_date,
            "target_week_end": target_end,
            "target_year": target_year,
            "target_month": int(target_end.month),
            "target_week": int(target_end.isocalendar().week),
            "spy_return_next_week": target_close / decision_close - 1.0,
            "month_number": float(target_end.month),
            "quarter_number": float(((target_end.month - 1) // 3) + 1),
            "calendar_week": float(target_end.isocalendar().week),
            "locked_period": bool(target_end >= LOCKED_START),
            "sp500_down_year": bool(target_year in down_years),
            **_political_features(target_year),
        }
        for name, value in feature_row.items():
            row[name] = float(value) if pd.notna(value) else np.nan
        rows.append(row)
    examples = pd.DataFrame(rows)
    if examples.empty:
        return examples
    examples["decision_date"] = pd.to_datetime(examples["decision_date"])
    examples["target_week_end"] = pd.to_datetime(examples["target_week_end"])
    examples["target_year"] = examples["target_year"].astype(int)
    examples["target_month"] = examples["target_month"].astype(int)
    examples["target_week"] = examples["target_week"].astype(int)
    weekly_closes = _weekly_asset_closes(data)
    extra_columns: dict[str, np.ndarray] = {}
    for asset, close in weekly_closes.items():
        next_return = close.shift(-1) / close - 1.0
        extra_columns[_asset_col(asset, "return_next_week")] = _align_weekly_series(examples, next_return)
        for weeks in (4, 13, 26, 52):
            momentum = close / close.shift(weeks) - 1.0
            extra_columns[_asset_col(asset, f"mom_{weeks}w")] = _align_weekly_series(examples, momentum)
        weekly_return = close.pct_change()
        vol = weekly_return.rolling(13).std() * np.sqrt(52.0)
        extra_columns[_asset_col(asset, "vol_13w")] = _align_weekly_series(examples, vol)
    if extra_columns:
        examples = pd.concat([examples, pd.DataFrame(extra_columns, index=examples.index)], axis=1)
    return examples.replace([np.inf, -np.inf], np.nan)


def evaluate_weekly_multi_asset_candidate(
    examples: pd.DataFrame,
    candidate: WeeklyMultiAssetCandidate,
    *,
    method: str,
    score_mode: str = WEEKLY_DOWN_5PCT_SCORE_MODE,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    usable = _without_locked_weekly(examples)
    exposure_candidate = MonthlyRiskCandidate(
        specs=candidate.specs,
        intercept=candidate.intercept,
        scale=candidate.scale,
        smoothing=candidate.smoothing,
    )
    exposures = _candidate_exposure(usable, exposure_candidate)
    selected_assets = _select_assets(usable, candidate)
    asset_returns = _selected_asset_returns(usable, selected_assets)
    strategy_returns = exposures * asset_returns
    spy_returns = pd.to_numeric(usable["spy_return_next_week"], errors="coerce").fillna(0.0).to_numpy()
    candidate_id = _candidate_id(candidate)
    positions = pd.DataFrame(
        {
            "candidate_id": candidate_id,
            "method": method,
            "decision_date": pd.to_datetime(usable["decision_date"]).to_numpy(),
            "target_week_end": pd.to_datetime(usable["target_week_end"]).to_numpy(),
            "target_year": usable["target_year"].astype(int).to_numpy(),
            "target_week": usable["target_week"].astype(int).to_numpy(),
            "period": _period_labels_weekly(usable),
            "traded_asset": selected_assets,
            "asset_return": asset_returns,
            "spy_return": spy_returns,
            "sp500_down_year": usable["sp500_down_year"].astype(bool).to_numpy(),
            "exposure": exposures,
            "strategy_return": strategy_returns,
        }
    )
    year_by_year = _year_by_year(positions)
    train_metrics = _weekly_return_metrics(positions.loc[positions["period"] == "train", "strategy_return"])
    validation_metrics = _weekly_return_metrics(
        positions.loc[positions["period"] == "validation", "strategy_return"]
    )
    train_years = year_by_year[year_by_year["period"] == "train"]
    validation_years = year_by_year[year_by_year["period"] == "validation"]
    train_down = train_years[train_years["sp500_down_year"].astype(bool)]
    validation_down = validation_years[validation_years["sp500_down_year"].astype(bool)]
    feature_names = _candidate_features(exposure_candidate)
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "method": method,
        "specs": ";".join(candidate.specs),
        "rules": _rules_text(exposure_candidate),
        "features": ",".join(feature_names),
        "feature_count": len(feature_names),
        "assets": ",".join(candidate.assets),
        "asset_selector": candidate.selector,
        "traded_asset_mode": "weekly_rotation",
        "weekly_exposure_formula": _formula_text(candidate),
        "intercept": float(candidate.intercept),
        "scale": float(candidate.scale),
        "smoothing": float(candidate.smoothing),
        "train_years_positive": _count_positive_years(train_years),
        "train_years_total": int(len(train_years)),
        "validation_years_positive": _count_positive_years(validation_years),
        "validation_years_total": int(len(validation_years)),
        "train_down_years_ge_5pct": _count_down_years_ge_5pct(train_down),
        "train_down_years_total": int(len(train_down)),
        "validation_down_years_ge_5pct": _count_down_years_ge_5pct(validation_down),
        "validation_down_years_total": int(len(validation_down)),
        "train_min_year_return": _min_year_return(train_years),
        "validation_min_year_return": _min_year_return(validation_years),
        "train_down_min_return": _min_year_return(train_down),
        "validation_down_min_return": _min_year_return(validation_down),
        "train_cagr": train_metrics["cagr"],
        "validation_cagr": validation_metrics["cagr"],
        "train_sharpe": train_metrics["sharpe"],
        "validation_sharpe": validation_metrics["sharpe"],
        "train_mdd": train_metrics["mdd"],
        "validation_mdd": validation_metrics["mdd"],
        "train_calmar": train_metrics["calmar"],
        "validation_calmar": validation_metrics["calmar"],
        "average_exposure": float(np.mean(exposures)) if len(exposures) else 0.0,
        "average_abs_exposure": float(np.mean(np.abs(exposures))) if len(exposures) else 0.0,
        "min_exposure": float(np.min(exposures)) if len(exposures) else 0.0,
        "max_exposure": float(np.max(exposures)) if len(exposures) else 0.0,
        "weeks_long": int(np.sum(exposures > 0.05)),
        "weeks_cash_like": int(np.sum(np.abs(exposures) <= 0.05)),
        "weeks_short": int(np.sum(exposures < -0.05)),
        "exposure_turnover": float(np.mean(np.abs(np.diff(exposures)))) if len(exposures) > 1 else 0.0,
        "unique_assets_used": int(pd.Series(selected_assets).nunique()) if len(selected_assets) else 0,
        "most_used_asset": str(pd.Series(selected_assets).mode().iloc[0]) if len(selected_assets) else "",
        "cash_allowed": True,
        "short_allowed": True,
        "locked_opened": False,
        "locked_weeks": 0,
        "validation_role": "report_only",
        "score_mode": score_mode,
        "accepted": False,
        "verified_train_validation_5pct": False,
        "train_calmar_gt_1": False,
        "validation_calmar_gt_1": False,
        "validation_calmar_ratio_to_train": np.nan,
        "validation_calmar_ge_80pct_train": False,
        "verified_calmar_similarity": False,
        "train_sharpe_gt_1": False,
        "validation_sharpe_gt_1": False,
        "validation_sharpe_ratio_to_train": np.nan,
        "validation_sharpe_ge_80pct_train": False,
        "train_cagr_ge_4pct": False,
        "validation_cagr_ge_3pct": False,
        "exposure_active_enough": False,
        "verified_sharpe_robust": False,
        "rejection_reason": "",
        "weekly_multi_asset_score": 0.0,
    }
    if score_mode == WEEKLY_MAX_CALMAR_SCORE_MODE:
        _stamp_calmar_verification(row)
        row["rejection_reason"] = _calmar_rejection_reason(row)
        row["accepted"] = row["rejection_reason"] == ""
        row["verified_train_validation_5pct"] = False
        row["weekly_multi_asset_score"] = _weekly_calmar_score(row)
    elif score_mode == WEEKLY_MAX_SHARPE_SCORE_MODE:
        _stamp_sharpe_verification(row)
        row["rejection_reason"] = _sharpe_rejection_reason(row)
        row["accepted"] = row["rejection_reason"] == ""
        row["verified_train_validation_5pct"] = False
        row["weekly_multi_asset_score"] = _weekly_sharpe_score(row)
    else:
        row["rejection_reason"] = _rejection_reason(row)
        row["accepted"] = row["rejection_reason"] == ""
        row["verified_train_validation_5pct"] = _is_verified(row)
        row["weekly_multi_asset_score"] = _weekly_score(row)
    return row, positions, year_by_year


def evaluate_weekly_machine_learning_candidate(
    examples: pd.DataFrame,
    candidate: WeeklyMachineLearningCandidate,
    *,
    method: str = "machine_learning",
    score_mode: str = WEEKLY_MAX_SHARPE_SCORE_MODE,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    usable = _without_locked_weekly(examples)
    candidate_id = _ml_candidate_id(candidate)
    selected_assets = _select_assets(
        usable,
        WeeklyMultiAssetCandidate(tuple(), candidate.assets, selector=candidate.selector),
    )
    asset_returns = _selected_asset_returns(usable, selected_assets)
    periods = _period_labels_weekly(usable)
    train_mask = periods == "train"
    features = [feature for feature in candidate.features if feature in usable.columns]
    if not features:
        features = _ml_feature_names(usable)[:1]
    x_raw = usable[features].apply(pd.to_numeric, errors="coerce")
    train_x = x_raw.loc[train_mask]
    medians = train_x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x_raw.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0).to_numpy(dtype=float)
    y = np.asarray(asset_returns, dtype=float)
    if train_mask.sum() < 20 or np.nanstd(y[train_mask]) <= 1e-12:
        predictions = np.zeros(len(usable), dtype=float)
        model_error = "insufficient_train_data"
    else:
        try:
            predictions = _fit_predict_ml(candidate, x[train_mask], y[train_mask], x)
            model_error = ""
        except Exception as exc:
            predictions = np.zeros(len(usable), dtype=float)
            model_error = f"{type(exc).__name__}: {exc}"
    train_predictions = predictions[train_mask] if len(predictions) else np.array([], dtype=float)
    pred_std = float(np.nanstd(train_predictions)) if len(train_predictions) else 0.0
    denom = pred_std if pred_std > 1e-12 else 1.0
    exposures = np.clip(np.tanh(candidate.intercept + candidate.scale * predictions / denom), -1.0, 1.0)
    strategy_returns = exposures * asset_returns
    spy_returns = pd.to_numeric(usable["spy_return_next_week"], errors="coerce").fillna(0.0).to_numpy()
    positions = pd.DataFrame(
        {
            "candidate_id": candidate_id,
            "method": method,
            "decision_date": pd.to_datetime(usable["decision_date"]).to_numpy(),
            "target_week_end": pd.to_datetime(usable["target_week_end"]).to_numpy(),
            "target_year": usable["target_year"].astype(int).to_numpy(),
            "target_week": usable["target_week"].astype(int).to_numpy(),
            "period": periods,
            "traded_asset": selected_assets,
            "asset_return": asset_returns,
            "spy_return": spy_returns,
            "sp500_down_year": usable["sp500_down_year"].astype(bool).to_numpy(),
            "exposure": exposures,
            "strategy_return": strategy_returns,
            "ml_prediction": predictions,
        }
    )
    year_by_year = _year_by_year(positions)
    train_metrics = _weekly_return_metrics(positions.loc[positions["period"] == "train", "strategy_return"])
    validation_metrics = _weekly_return_metrics(
        positions.loc[positions["period"] == "validation", "strategy_return"]
    )
    train_years = year_by_year[year_by_year["period"] == "train"]
    validation_years = year_by_year[year_by_year["period"] == "validation"]
    train_down = train_years[train_years["sp500_down_year"].astype(bool)]
    validation_down = validation_years[validation_years["sp500_down_year"].astype(bool)]
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "method": method,
        "specs": ";".join(features),
        "rules": f"ML {candidate.model} fit on train only, predict weekly exposure",
        "features": ",".join(features),
        "feature_count": len(features),
        "assets": ",".join(candidate.assets),
        "asset_selector": candidate.selector,
        "traded_asset_mode": "weekly_rotation_ml",
        "weekly_exposure_formula": f"asset = weekly best by {candidate.selector}; exposure = tanh({candidate.intercept:.4f} + {candidate.scale:.4f} * {candidate.model}_prediction_z)",
        "intercept": float(candidate.intercept),
        "scale": float(candidate.scale),
        "smoothing": 0.0,
        "ml_model": candidate.model,
        "ml_alpha": float(candidate.alpha),
        "ml_n_estimators": int(candidate.n_estimators),
        "ml_max_depth": int(candidate.max_depth),
        "ml_learning_rate": float(candidate.learning_rate),
        "ml_random_seed": int(candidate.random_seed),
        "ml_model_error": model_error,
        "train_years_positive": _count_positive_years(train_years),
        "train_years_total": int(len(train_years)),
        "validation_years_positive": _count_positive_years(validation_years),
        "validation_years_total": int(len(validation_years)),
        "train_down_years_ge_5pct": _count_down_years_ge_5pct(train_down),
        "train_down_years_total": int(len(train_down)),
        "validation_down_years_ge_5pct": _count_down_years_ge_5pct(validation_down),
        "validation_down_years_total": int(len(validation_down)),
        "train_min_year_return": _min_year_return(train_years),
        "validation_min_year_return": _min_year_return(validation_years),
        "train_down_min_return": _min_year_return(train_down),
        "validation_down_min_return": _min_year_return(validation_down),
        "train_cagr": train_metrics["cagr"],
        "validation_cagr": validation_metrics["cagr"],
        "train_sharpe": train_metrics["sharpe"],
        "validation_sharpe": validation_metrics["sharpe"],
        "train_mdd": train_metrics["mdd"],
        "validation_mdd": validation_metrics["mdd"],
        "train_calmar": train_metrics["calmar"],
        "validation_calmar": validation_metrics["calmar"],
        "average_exposure": float(np.mean(exposures)) if len(exposures) else 0.0,
        "average_abs_exposure": float(np.mean(np.abs(exposures))) if len(exposures) else 0.0,
        "min_exposure": float(np.min(exposures)) if len(exposures) else 0.0,
        "max_exposure": float(np.max(exposures)) if len(exposures) else 0.0,
        "weeks_long": int(np.sum(exposures > 0.05)),
        "weeks_cash_like": int(np.sum(np.abs(exposures) <= 0.05)),
        "weeks_short": int(np.sum(exposures < -0.05)),
        "exposure_turnover": float(np.mean(np.abs(np.diff(exposures)))) if len(exposures) > 1 else 0.0,
        "unique_assets_used": int(pd.Series(selected_assets).nunique()) if len(selected_assets) else 0,
        "most_used_asset": str(pd.Series(selected_assets).mode().iloc[0]) if len(selected_assets) else "",
        "cash_allowed": True,
        "short_allowed": True,
        "locked_opened": False,
        "locked_weeks": 0,
        "validation_role": "report_only",
        "score_mode": score_mode,
        "accepted": False,
        "verified_train_validation_5pct": False,
        "train_calmar_gt_1": False,
        "validation_calmar_gt_1": False,
        "validation_calmar_ratio_to_train": np.nan,
        "validation_calmar_ge_80pct_train": False,
        "verified_calmar_similarity": False,
        "train_sharpe_gt_1": False,
        "validation_sharpe_gt_1": False,
        "validation_sharpe_ratio_to_train": np.nan,
        "validation_sharpe_ge_80pct_train": False,
        "train_cagr_ge_4pct": False,
        "validation_cagr_ge_3pct": False,
        "exposure_active_enough": False,
        "verified_sharpe_robust": False,
        "rejection_reason": "",
        "weekly_multi_asset_score": 0.0,
    }
    _stamp_sharpe_verification(row)
    row["rejection_reason"] = _sharpe_rejection_reason(row)
    row["accepted"] = row["rejection_reason"] == ""
    row["weekly_multi_asset_score"] = _weekly_sharpe_score(row)
    return row, positions, year_by_year


def run_weekly_multi_asset_search(
    examples: pd.DataFrame,
    config: MonthlyRiskSearchConfig,
    *,
    method: str,
) -> list[dict[str, object]]:
    if method not in WEEKLY_METHODS:
        raise ValueError(f"unknown weekly method: {method}")
    catalog = _build_weekly_spec_catalog(examples)
    assets = _available_assets_from_examples(examples)
    if not catalog or not assets:
        return []
    rng = np.random.default_rng(config.random_seed + config.stage + _method_offset(method))
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str, float, float, float]] = set()
    if method == "random_broad":
        rows = _run_random_broad(examples, catalog, assets, config=config, rng=rng, seen=seen)
    elif method == "beam":
        rows = _run_beam(examples, catalog, assets, config=config, rng=rng, seen=seen, method=method)
    elif method == "genetic":
        rows = _run_genetic(examples, catalog, assets, config=config, rng=rng, seen=seen)
    elif method == "bayesian_like":
        rows = _run_bayesian_like(examples, catalog, assets, config=config, rng=rng, seen=seen)
    else:
        rows = _run_bandit(examples, catalog, assets, config=config, rng=rng, seen=seen)
    sorted_rows = sorted(rows, key=lambda row: float(row["weekly_multi_asset_score"]), reverse=True)
    if config.top_rows_per_stage > 0:
        accepted = [row for row in sorted_rows if bool(row.get("accepted")) or bool(row.get("verified_train_validation_5pct"))]
        top = sorted_rows[: config.top_rows_per_stage]
        by_id = {str(row["candidate_id"]): row for row in [*top, *accepted]}
        sorted_rows = sorted(by_id.values(), key=lambda row: float(row["weekly_multi_asset_score"]), reverse=True)
    return sorted_rows


def write_weekly_multi_asset_outputs(
    rows: list[dict[str, object]],
    examples: pd.DataFrame,
    output_dir: str | Path,
    *,
    method: str,
    stage: int | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.DataFrame(rows)
    leaderboard.to_csv(output / "weekly_multi_asset_sp500_down_5pct_leaderboard.csv", index=False)
    if stage is not None:
        leaderboard.to_csv(output / f"weekly_multi_asset_sp500_down_5pct_leaderboard_stage_{method}_{stage}.csv", index=False)
    with (output / "weekly_multi_asset_sp500_down_5pct_candidates.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
    summary = _summary(rows, method=method, stage=stage)
    (output / "weekly_multi_asset_sp500_down_5pct_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if rows:
        best = _candidate_from_row(rows[0])
        _, positions, year_by_year = evaluate_weekly_multi_asset_candidate(examples, best, method=method)
    else:
        positions = pd.DataFrame()
        year_by_year = pd.DataFrame()
    positions.to_csv(output / "weekly_multi_asset_sp500_down_5pct_weekly_positions.csv", index=False)
    year_by_year.to_csv(output / "weekly_multi_asset_sp500_down_5pct_year_by_year.csv", index=False)


def merge_weekly_multi_asset_leaderboards(
    paths: list[str | Path],
    output_dir: str | Path,
    *,
    examples: pd.DataFrame | None = None,
    progress_every: int = 0,
    max_output_rows: int = 0,
) -> dict[str, object]:
    frames = []
    existing_paths = [Path(path) for path in paths if Path(path).exists()]
    for index, path in enumerate(existing_paths, start=1):
        frames.append(pd.read_csv(path))
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == len(existing_paths)):
            print(f"read {index}/{len(existing_paths)} leaderboard files", flush=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not frames:
        empty = pd.DataFrame()
        empty.to_csv(output / "weekly_multi_asset_sp500_down_5pct_leaderboard.csv", index=False)
        empty.to_csv(output / "weekly_multi_asset_sp500_down_5pct_verified.csv", index=False)
        empty.to_csv(output / "weekly_multi_asset_sp500_down_5pct_methods.csv", index=False)
        empty.to_csv(output / "weekly_multi_asset_sp500_down_5pct_year_by_year.csv", index=False)
        empty.to_csv(output / "weekly_multi_asset_sp500_down_5pct_weekly_positions.csv", index=False)
        summary = _empty_summary()
        (output / "weekly_multi_asset_sp500_down_5pct_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary
    print(f"concatenating {len(frames)} leaderboard frames", flush=True)
    raw = pd.concat(frames, ignore_index=True)
    print(f"raw rows: {len(raw)}", flush=True)
    methods = _method_summary(raw)
    methods.to_csv(output / "weekly_multi_asset_sp500_down_5pct_methods.csv", index=False)
    print("deduplicating candidates", flush=True)
    merged = (
        raw.sort_values("weekly_multi_asset_score", ascending=False)
        .drop_duplicates("candidate_id", keep="first")
        .sort_values("weekly_multi_asset_score", ascending=False)
    )
    print(f"merged rows: {len(merged)}", flush=True)
    verified = _verified(merged)
    print(f"verified rows: {len(verified)}", flush=True)
    output_merged = _limit_weekly_output_rows(merged, verified, max_output_rows)
    print(f"output leaderboard rows: {len(output_merged)}", flush=True)
    output_merged.to_csv(output / "weekly_multi_asset_sp500_down_5pct_leaderboard.csv", index=False)
    verified.to_csv(output / "weekly_multi_asset_sp500_down_5pct_verified.csv", index=False)
    if examples is not None and not merged.empty:
        print("building best-candidate positions", flush=True)
        method = str(merged.iloc[0].get("method", "merged") or "merged")
        score_mode = str(merged.iloc[0].get("score_mode", WEEKLY_DOWN_5PCT_SCORE_MODE) or WEEKLY_DOWN_5PCT_SCORE_MODE)
        if method == "machine_learning" or str(merged.iloc[0].get("ml_model", "")):
            best_ml = _ml_candidate_from_row(merged.iloc[0])
            _, positions, year_by_year = evaluate_weekly_machine_learning_candidate(
                examples,
                best_ml,
                method=method,
                score_mode=score_mode,
            )
        else:
            best = _candidate_from_row(merged.iloc[0])
            _, positions, year_by_year = evaluate_weekly_multi_asset_candidate(
                examples,
                best,
                method=method,
                score_mode=score_mode,
            )
    else:
        positions = pd.DataFrame()
        year_by_year = pd.DataFrame()
    positions.to_csv(output / "weekly_multi_asset_sp500_down_5pct_weekly_positions.csv", index=False)
    year_by_year.to_csv(output / "weekly_multi_asset_sp500_down_5pct_year_by_year.csv", index=False)
    summary = {
        "rows": int(len(merged)),
        "candidates_evaluated": int(len(raw)),
        "accepted": int(merged.get("accepted", pd.Series(dtype=bool)).astype(bool).sum()),
        "verified_train_validation_5pct": int(merged.get("verified_train_validation_5pct", pd.Series(dtype=bool)).astype(bool).sum()),
        "verified_calmar_similarity": int(merged.get("verified_calmar_similarity", pd.Series(dtype=bool)).astype(bool).sum()),
        "verified_sharpe_robust": int(merged.get("verified_sharpe_robust", pd.Series(dtype=bool)).astype(bool).sum()),
        "unique_verified_train_validation_5pct": int(verified["candidate_id"].nunique()) if "candidate_id" in verified else 0,
        "best_candidate": str(merged.iloc[0]["candidate_id"]) if not merged.empty else None,
        "best_method": str(merged.iloc[0].get("method", "")) if not merged.empty else None,
        "best_assets": str(merged.iloc[0].get("assets", "")) if not merged.empty else None,
        "best_asset_selector": str(merged.iloc[0].get("asset_selector", "")) if not merged.empty else None,
        "best_train_min_year_return": float(merged.iloc[0].get("train_min_year_return", np.nan)) if not merged.empty else None,
        "best_validation_min_year_return": float(merged.iloc[0].get("validation_min_year_return", np.nan)) if not merged.empty else None,
        "best_train_down_min_return": float(merged.iloc[0].get("train_down_min_return", np.nan)) if not merged.empty else None,
        "best_validation_down_min_return": float(merged.iloc[0].get("validation_down_min_return", np.nan)) if not merged.empty else None,
        "best_verified_candidate": str(verified.iloc[0]["candidate_id"]) if not verified.empty else None,
        "locked_opened": False,
        "score_mode": str(merged.iloc[0].get("score_mode", WEEKLY_DOWN_5PCT_SCORE_MODE)) if not merged.empty else WEEKLY_DOWN_5PCT_SCORE_MODE,
        "validation_role": "report_only",
        "train_down_years_expected": [2000, 2001, 2002],
        "validation_down_years_expected": [2008, 2018],
        "tradable_assets": list(TRADABLE_ASSETS),
        "methods": list(WEEKLY_METHODS),
    }
    (output / "weekly_multi_asset_sp500_down_5pct_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _limit_weekly_output_rows(merged: pd.DataFrame, verified: pd.DataFrame, max_output_rows: int) -> pd.DataFrame:
    if max_output_rows <= 0 or len(merged) <= max_output_rows:
        return merged
    keep = merged.head(max_output_rows)
    if verified.empty or "candidate_id" not in verified:
        return keep
    combined = pd.concat([keep, verified], ignore_index=True)
    return (
        combined.sort_values("weekly_multi_asset_score", ascending=False)
        .drop_duplicates("candidate_id", keep="first")
        .sort_values("weekly_multi_asset_score", ascending=False)
    )


def _run_random_broad(
    examples: pd.DataFrame,
    catalog: list[str],
    assets: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str, float, float, float]],
) -> list[dict[str, object]]:
    candidates = _seed_candidates(catalog, assets, config=config, rng=rng, method="random_broad")
    extra = [_random_candidate(catalog, assets, config=config, rng=rng) for _ in range(config.seed_pool)]
    return _evaluate_unique(examples, [*candidates, *extra], seen, method="random_broad", score_mode=config.score_mode)


def _run_beam(
    examples: pd.DataFrame,
    catalog: list[str],
    assets: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str, float, float, float]],
    method: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seed_rows = _evaluate_unique(
        examples,
        _seed_candidates(catalog, assets, config=config, rng=rng, method=method),
        seen,
        method=method,
        score_mode=config.score_mode,
    )
    rows.extend(seed_rows)
    beam = _select_beam(seed_rows, config.beam_width)
    for _ in range(config.generations):
        children: list[WeeklyMultiAssetCandidate] = []
        for parent in beam:
            candidate = _candidate_from_row(parent)
            for _ in range(config.mutations_per_parent):
                children.append(_mutate_candidate(candidate, catalog, assets, config=config, rng=rng))
        child_rows = _evaluate_unique(examples, children, seen, method=method, score_mode=config.score_mode)
        rows.extend(child_rows)
        beam = _select_beam([*beam, *child_rows], config.beam_width)
    return rows


def _run_genetic(
    examples: pd.DataFrame,
    catalog: list[str],
    assets: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str, float, float, float]],
) -> list[dict[str, object]]:
    rows = _evaluate_unique(
        examples,
        _seed_candidates(catalog, assets, config=config, rng=rng, method="genetic"),
        seen,
        method="genetic",
        score_mode=config.score_mode,
    )
    population = _select_beam(rows, max(config.beam_width, 20))
    for _ in range(config.generations):
        parents = [_candidate_from_row(row) for row in population]
        children: list[WeeklyMultiAssetCandidate] = []
        for _ in range(config.beam_width * config.mutations_per_parent):
            left = parents[int(rng.integers(0, len(parents)))]
            right = parents[int(rng.integers(0, len(parents)))]
            children.append(_mutate_candidate(_crossover(left, right, rng), catalog, assets, config=config, rng=rng))
        child_rows = _evaluate_unique(examples, children, seen, method="genetic", score_mode=config.score_mode)
        rows.extend(child_rows)
        population = _select_beam([*population, *child_rows], max(config.beam_width, 20))
    return rows


def _run_bayesian_like(
    examples: pd.DataFrame,
    catalog: list[str],
    assets: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str, float, float, float]],
) -> list[dict[str, object]]:
    rows = _evaluate_unique(
        examples,
        _seed_candidates(catalog, assets, config=config, rng=rng, method="bayesian_like"),
        seen,
        method="bayesian_like",
        score_mode=config.score_mode,
    )
    for _ in range(config.generations):
        weights = _spec_weights_from_rows(rows, catalog)
        candidates = [_weighted_candidate(catalog, weights, assets, config=config, rng=rng) for _ in range(config.beam_width * config.mutations_per_parent)]
        child_rows = _evaluate_unique(examples, candidates, seen, method="bayesian_like", score_mode=config.score_mode)
        rows.extend(child_rows)
    return rows


def _run_bandit(
    examples: pd.DataFrame,
    catalog: list[str],
    assets: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str, float, float, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups = _catalog_groups(catalog)
    for _ in range(config.generations + 1):
        rewards = _group_rewards(rows, groups)
        candidates: list[WeeklyMultiAssetCandidate] = []
        for _ in range(max(config.seed_pool // max(config.generations + 1, 1), config.beam_width)):
            group = _choose_group(rewards, rng)
            group_catalog = groups.get(group, catalog) or catalog
            candidates.append(_random_candidate(group_catalog, assets, config=config, rng=rng))
        rows.extend(_evaluate_unique(examples, candidates, seen, method="bandit", score_mode=config.score_mode))
    return rows


def _weekly_asset_closes(data: pd.DataFrame) -> dict[str, pd.Series]:
    closes = {"SPY": pd.to_numeric(data["close"], errors="coerce")}
    for asset in TRADABLE_ASSETS:
        if asset == "SPY":
            continue
        ratio_column = _ratio_column(asset)
        if ratio_column not in data:
            continue
        ratio = pd.to_numeric(data[ratio_column], errors="coerce")
        close = closes["SPY"] * ratio
        if close.notna().sum() >= 104:
            closes[asset] = close
    return {asset: close.resample("W-FRI").last() for asset, close in closes.items()}


def _align_weekly_series(examples: pd.DataFrame, series: pd.Series) -> np.ndarray:
    lookup = series.copy()
    lookup.index = pd.to_datetime(lookup.index)
    return lookup.reindex(pd.to_datetime(examples["decision_date"])).to_numpy(dtype=float)


def _benchmark_down_years(data: pd.DataFrame, benchmark_daily: pd.DataFrame | None) -> set[int]:
    spy_close = pd.to_numeric(data["close"], errors="coerce").dropna()
    sp500_close = spy_close
    if benchmark_daily is not None and not benchmark_daily.empty:
        benchmark = _normalize_daily(benchmark_daily)
        sp500_close = pd.to_numeric(benchmark["close"], errors="coerce").dropna()
    spy_year = spy_close.resample("YE").last().pct_change()
    sp500_year = sp500_close.resample("YE").last().pct_change()
    years: set[int] = set()
    for date, spy_return in spy_year.dropna().items():
        year = int(date.year)
        sp500_matches = sp500_year[sp500_year.index.year == year]
        if sp500_matches.empty:
            continue
        if float(spy_return) < 0.0 and float(sp500_matches.iloc[0]) < 0.0:
            years.add(year)
    return years


def _available_assets_from_examples(examples: pd.DataFrame) -> list[str]:
    assets = []
    for asset in TRADABLE_ASSETS:
        column = _asset_col(asset, "return_next_week")
        if column in examples and pd.to_numeric(examples[column], errors="coerce").notna().sum() >= 104:
            assets.append(asset)
    return assets


def _asset_universes(assets: list[str]) -> list[tuple[str, ...]]:
    groups = [
        ("SPY",),
        ("SPY", "QQQ", "IWM", "RSP", "DIA"),
        ("SPY", "EFA", "EEM", "EWJ", "EWG", "EWU", "FXI"),
        ("SPY", "SHY", "IEI", "IEF", "TLT", "LQD", "HYG"),
        ("SPY", "GLD", "SLV", "DBC", "USO", "CPER", "UUP"),
        ("SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "XLI", "XLB", "XLRE", "XLC"),
        tuple(TRADABLE_ASSETS),
    ]
    available = set(assets)
    universes: list[tuple[str, ...]] = []
    for group in groups:
        filtered = tuple(asset for asset in group if asset in available)
        if filtered and filtered not in universes:
            universes.append(filtered)
    for asset in assets:
        single = (asset,)
        if single not in universes:
            universes.append(single)
    return universes


def _select_assets(examples: pd.DataFrame, candidate: WeeklyMultiAssetCandidate) -> np.ndarray:
    assets = [asset for asset in candidate.assets if _asset_col(asset, "return_next_week") in examples]
    if not assets:
        assets = ["SPY"]
    score_suffix = {
        "momentum_4w": "mom_4w",
        "momentum_13w": "mom_13w",
        "momentum_26w": "mom_26w",
        "momentum_52w": "mom_52w",
        "low_vol_13w": "vol_13w",
    }.get(candidate.selector, "mom_26w")
    score_columns: list[np.ndarray] = []
    for asset in assets:
        column = _asset_col(asset, score_suffix)
        values = pd.to_numeric(examples.get(column, pd.Series(index=examples.index, dtype=float)), errors="coerce")
        if candidate.selector == "low_vol_13w":
            values = -values
        score_columns.append(values.to_numpy(dtype=float))
    scores = np.column_stack(score_columns)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    all_missing = np.isneginf(scores).all(axis=1)
    selected_index = np.argmax(scores, axis=1)
    selected = np.array([assets[index] for index in selected_index], dtype=object)
    selected[all_missing] = "SPY"
    return selected


def _selected_asset_returns(examples: pd.DataFrame, selected_assets: np.ndarray) -> np.ndarray:
    returns = np.zeros(len(examples), dtype=float)
    for asset in pd.unique(selected_assets):
        column = _asset_col(str(asset), "return_next_week")
        if column in examples:
            values = pd.to_numeric(examples[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            mask = selected_assets == asset
            returns[mask] = values[mask]
    return returns


def _target_dates_weekly(examples: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(examples["target_week_end"])


def _without_locked_weekly(examples: pd.DataFrame) -> pd.DataFrame:
    return examples.loc[_target_dates_weekly(examples) < LOCKED_START].copy()


def _period_labels_weekly(examples: pd.DataFrame) -> np.ndarray:
    dates = _target_dates_weekly(examples)
    labels = np.where(dates <= TRAIN_END, "train", np.where(dates <= VALIDATION_END, "validation", "locked"))
    return labels.astype(object)


def _year_by_year(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (period, year), group in positions.groupby(["period", "target_year"], sort=True):
        strategy_return = float((1.0 + group["strategy_return"]).prod() - 1.0)
        spy_return = float((1.0 + group["spy_return"]).prod() - 1.0)
        rows.append(
            {
                "candidate_id": str(group["candidate_id"].iloc[0]),
                "method": str(group["method"].iloc[0]),
                "period": period,
                "year": int(year),
                "strategy_return": strategy_return,
                "spy_return": spy_return,
                "excess_return": strategy_return - spy_return,
                "sp500_down_year": bool(group["sp500_down_year"].any()),
                "down_year_pass_5pct": bool(group["sp500_down_year"].any() and strategy_return >= MIN_DOWN_YEAR_RETURN),
                "positive_year": bool(strategy_return > 0.0),
                "average_exposure": float(group["exposure"].mean()),
                "min_exposure": float(group["exposure"].min()),
                "max_exposure": float(group["exposure"].max()),
                "dominant_asset": str(group["traded_asset"].mode().iloc[0]),
                "unique_assets_used": int(group["traded_asset"].nunique()),
                "weeks": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _weekly_return_metrics(returns: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(values) == 0:
        return {"cagr": 0.0, "mdd": 0.0, "calmar": 0.0, "sharpe": 0.0}
    equity = np.cumprod(1.0 + values)
    years = max(len(values) / 52.0, 1e-9)
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0.0 else -1.0
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    mdd = float(drawdown.min())
    calmar = float(cagr / abs(mdd)) if mdd < 0.0 else float("inf") if cagr > 0 else 0.0
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sharpe = float(np.mean(values) / std * np.sqrt(52.0)) if std > 1e-12 else 0.0
    return {"cagr": cagr, "mdd": mdd, "calmar": calmar, "sharpe": sharpe}


def _build_weekly_spec_catalog(examples: pd.DataFrame) -> list[str]:
    usable = _without_locked_weekly(examples)
    train = usable.loc[_period_labels_weekly(usable) == "train"]
    if train.empty:
        return []
    skip = {
        "decision_date",
        "target_week_end",
        "target_year",
        "target_month",
        "target_week",
        "spy_return_next_week",
        "locked_period",
        "sp500_down_year",
    }
    skip.update({column for column in usable.columns if column.endswith("_return_next_week")})
    catalog: list[str] = []
    for column in usable.columns:
        if column in skip:
            continue
        series = pd.to_numeric(train[column], errors="coerce")
        if series.notna().sum() < max(52, int(len(train) * 0.20)):
            continue
        if series.nunique(dropna=True) < 2:
            continue
        quantiles = series.quantile([0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85]).dropna().drop_duplicates()
        for threshold in quantiles:
            for direction in (-1, 1):
                for weight in (1, 2, 3):
                    catalog.append(f"{column}|threshold|{float(threshold):.10g}|{direction}|{weight}")
        low_high = series.quantile([0.20, 0.80]).dropna()
        if len(low_high) == 2 and float(low_high.iloc[0]) < float(low_high.iloc[1]):
            for direction in (-1, 1):
                catalog.append(
                    f"{column}|range|{float(low_high.iloc[0]):.10g}|{float(low_high.iloc[1]):.10g}|{direction}|1"
                )
    return sorted(set(catalog))


def _seed_candidates(
    catalog: list[str],
    assets: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
    method: str,
) -> list[WeeklyMultiAssetCandidate]:
    stage_catalog = [spec for index, spec in enumerate(catalog) if index % config.total_stages == config.stage]
    if not stage_catalog:
        stage_catalog = catalog
    universes = _asset_universes(assets)
    candidates: list[WeeklyMultiAssetCandidate] = []
    exposures = (-0.85, -0.60, -0.35, -0.10, 0.0, 0.10, 0.35, 0.60, 0.85)
    scales = (0.25, 0.5, 0.75, 1.0, 1.25)
    selector_shift = _method_offset(method) % len(WEEKLY_ASSET_SELECTORS)
    for spec_index, spec in enumerate(stage_catalog[: config.seed_pool]):
        universe = universes[spec_index % len(universes)]
        selector = WEEKLY_ASSET_SELECTORS[(spec_index + selector_shift) % len(WEEKLY_ASSET_SELECTORS)]
        for scale in scales:
            candidates.append(WeeklyMultiAssetCandidate((spec,), universe, selector=selector, scale=scale))
        candidates.append(
            WeeklyMultiAssetCandidate((spec,), universe, selector=selector, intercept=float(rng.choice(exposures)), scale=0.35)
        )
    while len(candidates) < config.seed_pool:
        candidates.append(_random_candidate(stage_catalog, assets, config=config, rng=rng))
    return candidates[: config.seed_pool]


def _random_candidate(
    catalog: list[str],
    assets: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
) -> WeeklyMultiAssetCandidate:
    max_size = min(config.max_features, max(1, len(catalog)))
    size = int(rng.integers(1, max_size + 1))
    specs = tuple(sorted(rng.choice(catalog, size=size, replace=False).tolist()))
    universes = _asset_universes(assets)
    universe = universes[int(rng.integers(0, len(universes)))]
    return WeeklyMultiAssetCandidate(
        specs,
        tuple(sorted(set(universe))),
        selector=str(rng.choice(WEEKLY_ASSET_SELECTORS)),
        intercept=float(rng.uniform(-0.55, 0.55)),
        scale=float(rng.uniform(0.15, 1.45)),
        smoothing=float(rng.choice([0.0, 0.10, 0.20, 0.35, 0.50])),
    )


def _evaluate_unique(
    examples: pd.DataFrame,
    candidates: list[WeeklyMultiAssetCandidate],
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str, float, float, float]],
    *,
    method: str,
    score_mode: str = WEEKLY_DOWN_5PCT_SCORE_MODE,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        key = (
            tuple(sorted(candidate.specs)),
            tuple(sorted(candidate.assets)),
            candidate.selector,
            round(float(candidate.intercept), 4),
            round(float(candidate.scale), 4),
            round(float(candidate.smoothing), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        row, _, _ = evaluate_weekly_multi_asset_candidate(examples, candidate, method=method, score_mode=score_mode)
        rows.append(row)
    return rows


def _mutate_candidate(
    candidate: WeeklyMultiAssetCandidate,
    catalog: list[str],
    assets: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
) -> WeeklyMultiAssetCandidate:
    specs = list(candidate.specs)
    action = str(rng.choice(["add", "replace", "drop", "tweak", "selector", "assets"]))
    if action == "add" and len(specs) < config.max_features:
        specs.append(str(rng.choice(catalog)))
    elif action == "replace" and specs:
        specs[int(rng.integers(0, len(specs)))] = str(rng.choice(catalog))
    elif action == "drop" and len(specs) > 1:
        specs.pop(int(rng.integers(0, len(specs))))
    elif action == "selector":
        return WeeklyMultiAssetCandidate(tuple(sorted(set(specs))), candidate.assets, selector=str(rng.choice(WEEKLY_ASSET_SELECTORS)), intercept=candidate.intercept, scale=candidate.scale, smoothing=candidate.smoothing)
    elif action == "assets":
        universes = _asset_universes(assets)
        return WeeklyMultiAssetCandidate(tuple(sorted(set(specs))), tuple(sorted(set(universes[int(rng.integers(0, len(universes)))]))), selector=candidate.selector, intercept=candidate.intercept, scale=candidate.scale, smoothing=candidate.smoothing)
    else:
        specs = _tweak_spec_weight(specs, rng)
    return WeeklyMultiAssetCandidate(
        tuple(sorted(set(specs))),
        candidate.assets,
        selector=candidate.selector,
        intercept=float(np.clip(candidate.intercept + rng.normal(0.0, 0.15), -0.95, 0.95)),
        scale=float(np.clip(candidate.scale + rng.normal(0.0, 0.20), 0.05, 1.75)),
        smoothing=float(np.clip(candidate.smoothing + rng.normal(0.0, 0.10), 0.0, 0.80)),
    )


def _crossover(left: WeeklyMultiAssetCandidate, right: WeeklyMultiAssetCandidate, rng: np.random.Generator) -> WeeklyMultiAssetCandidate:
    specs = tuple(sorted(set(list(left.specs[: max(1, len(left.specs) // 2)]) + list(right.specs[len(right.specs) // 2 :]))))
    assets = left.assets if rng.random() < 0.5 else right.assets
    selector = left.selector if rng.random() < 0.5 else right.selector
    return WeeklyMultiAssetCandidate(
        specs or left.specs or right.specs,
        assets,
        selector=selector,
        intercept=float((left.intercept + right.intercept) / 2.0),
        scale=float((left.scale + right.scale) / 2.0),
        smoothing=float((left.smoothing + right.smoothing) / 2.0),
    )


def _tweak_spec_weight(specs: list[str], rng: np.random.Generator) -> list[str]:
    if not specs:
        return specs
    index = int(rng.integers(0, len(specs)))
    parts = specs[index].split("|")
    parts[-1] = str(int(rng.choice([1, 2, 3])))
    specs[index] = "|".join(parts)
    return specs


def _select_beam(rows: list[dict[str, object]], size: int) -> list[dict[str, object]]:
    by_id = {str(row["candidate_id"]): row for row in rows}
    return sorted(by_id.values(), key=lambda row: float(row["weekly_multi_asset_score"]), reverse=True)[:size]


def _spec_weights_from_rows(rows: list[dict[str, object]], catalog: list[str]) -> np.ndarray:
    scores = {spec: 1.0 for spec in catalog}
    for row in _select_beam(rows, 200):
        score = max(0.0, float(row.get("weekly_multi_asset_score", 0.0) or 0.0))
        for spec in str(row.get("specs", "")).split(";"):
            if spec in scores:
                scores[spec] += 1.0 + score / 1000.0
    weights = np.array([scores[spec] for spec in catalog], dtype=float)
    return weights / weights.sum() if weights.sum() > 0 else np.full(len(catalog), 1.0 / len(catalog))


def _weighted_candidate(
    catalog: list[str],
    weights: np.ndarray,
    assets: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
) -> WeeklyMultiAssetCandidate:
    max_size = min(config.max_features, max(1, len(catalog)))
    size = int(rng.integers(1, max_size + 1))
    specs = tuple(sorted(rng.choice(catalog, size=size, replace=False, p=weights).tolist()))
    universes = _asset_universes(assets)
    return WeeklyMultiAssetCandidate(
        specs,
        tuple(sorted(set(universes[int(rng.integers(0, len(universes)))]))),
        selector=str(rng.choice(WEEKLY_ASSET_SELECTORS)),
        intercept=float(rng.uniform(-0.55, 0.55)),
        scale=float(rng.uniform(0.15, 1.45)),
        smoothing=float(rng.choice([0.0, 0.10, 0.20, 0.35, 0.50])),
    )


def _catalog_groups(catalog: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for spec in catalog:
        feature = spec.split("|", 1)[0]
        group = feature.split("_", 1)[0]
        groups.setdefault(group, []).append(spec)
    return groups


def _group_rewards(rows: list[dict[str, object]], groups: dict[str, list[str]]) -> dict[str, float]:
    rewards = {group: 1.0 for group in groups}
    group_by_spec = {spec: group for group, specs in groups.items() for spec in specs}
    for row in _select_beam(rows, 300):
        score = max(0.0, float(row.get("weekly_multi_asset_score", 0.0) or 0.0))
        for spec in str(row.get("specs", "")).split(";"):
            group = group_by_spec.get(spec)
            if group:
                rewards[group] += 1.0 + score / 1500.0
    return rewards


def _choose_group(rewards: dict[str, float], rng: np.random.Generator) -> str:
    keys = list(rewards)
    values = np.array([max(0.01, rewards[key]) for key in keys], dtype=float)
    probs = values / values.sum()
    return str(rng.choice(keys, p=probs))


def _candidate_from_row(row: dict[str, object] | pd.Series) -> WeeklyMultiAssetCandidate:
    specs = tuple(part for part in str(row.get("specs", "")).split(";") if part)
    assets = tuple(part for part in str(row.get("assets", "SPY")).split(",") if part)
    return WeeklyMultiAssetCandidate(
        specs,
        assets or ("SPY",),
        selector=str(row.get("asset_selector", "momentum_26w") or "momentum_26w"),
        intercept=float(row.get("intercept", 0.0) or 0.0),
        scale=float(row.get("scale", 1.0) or 1.0),
        smoothing=float(row.get("smoothing", 0.0) or 0.0),
    )


def _ml_candidate_from_row(row: dict[str, object] | pd.Series) -> WeeklyMachineLearningCandidate:
    features = tuple(part for part in str(row.get("features", "")).split(",") if part)
    assets = tuple(part for part in str(row.get("assets", "SPY")).split(",") if part)
    return WeeklyMachineLearningCandidate(
        features=features,
        assets=assets or ("SPY",),
        selector=str(row.get("asset_selector", "momentum_26w") or "momentum_26w"),
        model=str(row.get("ml_model", "ridge") or "ridge"),
        alpha=float(row.get("ml_alpha", 1.0) or 1.0),
        n_estimators=int(float(row.get("ml_n_estimators", 64) or 64)),
        max_depth=int(float(row.get("ml_max_depth", 3) or 3)),
        learning_rate=float(row.get("ml_learning_rate", 0.05) or 0.05),
        scale=float(row.get("scale", 1.0) or 1.0),
        intercept=float(row.get("intercept", 0.0) or 0.0),
        random_seed=int(float(row.get("ml_random_seed", 0) or 0)),
    )


def _candidate_id(candidate: WeeklyMultiAssetCandidate) -> str:
    payload = json.dumps(
        {
            "specs": sorted(candidate.specs),
            "assets": sorted(candidate.assets),
            "selector": candidate.selector,
            "intercept": round(float(candidate.intercept), 6),
            "scale": round(float(candidate.scale), 6),
            "smoothing": round(float(candidate.smoothing), 6),
        },
        sort_keys=True,
    )
    return "weekly_multi_asset_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _formula_text(candidate: WeeklyMultiAssetCandidate) -> str:
    return (
        "asset = weekly best by "
        f"{candidate.selector}; exposure = clip({candidate.intercept:.4f} + "
        f"{candidate.scale:.4f} * weighted_rule_vote, -1, 1)"
    )


def _count_positive_years(years: pd.DataFrame) -> int:
    if years.empty:
        return 0
    return int((pd.to_numeric(years["strategy_return"], errors="coerce") > 0.0).sum())


def _count_down_years_ge_5pct(years: pd.DataFrame) -> int:
    if years.empty:
        return 0
    return int((pd.to_numeric(years["strategy_return"], errors="coerce") >= MIN_DOWN_YEAR_RETURN).sum())


def _min_year_return(years: pd.DataFrame) -> float:
    if years.empty:
        return float("nan")
    return float(pd.to_numeric(years["strategy_return"], errors="coerce").min())


def _rejection_reason(row: dict[str, object]) -> str:
    if int(row.get("train_years_total", 0) or 0) == 0:
        return "no_train_years"
    if int(row.get("train_down_years_total", 0) or 0) == 0:
        return "no_train_down_years"
    if int(row.get("train_down_years_ge_5pct", 0) or 0) < int(row.get("train_down_years_total", 0) or 0):
        return "train_down_year_below_5pct"
    if int(row.get("train_years_positive", 0) or 0) < int(row.get("train_years_total", 0) or 0):
        return "train_year_not_positive"
    return ""


def _is_verified(row: dict[str, object]) -> bool:
    return bool(row.get("accepted")) and (
        int(row.get("validation_years_total", 0) or 0) > 0
        and int(row.get("validation_down_years_total", 0) or 0) > 0
        and int(row.get("validation_years_positive", 0) or 0) == int(row.get("validation_years_total", 0) or 0)
        and int(row.get("validation_down_years_ge_5pct", 0) or 0) == int(row.get("validation_down_years_total", 0) or 0)
        and not bool(row.get("locked_opened"))
    )


def _stamp_calmar_verification(row: dict[str, object]) -> None:
    train_calmar = _finite_float(row.get("train_calmar"), default=-1.0)
    validation_calmar = _finite_float(row.get("validation_calmar"), default=-1.0)
    ratio = float("nan")
    if np.isfinite(train_calmar) and train_calmar > 0.0 and np.isfinite(validation_calmar):
        ratio = float(validation_calmar / train_calmar)
    row["train_calmar_gt_1"] = bool(np.isfinite(train_calmar) and train_calmar > 1.0)
    row["validation_calmar_gt_1"] = bool(np.isfinite(validation_calmar) and validation_calmar > 1.0)
    row["validation_calmar_ratio_to_train"] = ratio
    row["validation_calmar_ge_80pct_train"] = bool(np.isfinite(ratio) and ratio >= 0.80)
    row["verified_calmar_similarity"] = bool(
        row["train_calmar_gt_1"]
        and row["validation_calmar_gt_1"]
        and row["validation_calmar_ge_80pct_train"]
        and not bool(row.get("locked_opened"))
    )


def _stamp_sharpe_verification(row: dict[str, object]) -> None:
    train_sharpe = _finite_float(row.get("train_sharpe"), default=-1.0)
    validation_sharpe = _finite_float(row.get("validation_sharpe"), default=-1.0)
    train_cagr = _finite_float(row.get("train_cagr"), default=-1.0)
    validation_cagr = _finite_float(row.get("validation_cagr"), default=-1.0)
    avg_abs = _finite_float(row.get("average_abs_exposure"), default=0.0)
    ratio = float("nan")
    if np.isfinite(train_sharpe) and train_sharpe > 0.0 and np.isfinite(validation_sharpe):
        ratio = float(validation_sharpe / train_sharpe)
    row["train_sharpe_gt_1"] = bool(np.isfinite(train_sharpe) and train_sharpe > 1.0)
    row["validation_sharpe_gt_1"] = bool(np.isfinite(validation_sharpe) and validation_sharpe > 1.0)
    row["validation_sharpe_ratio_to_train"] = ratio
    row["validation_sharpe_ge_80pct_train"] = bool(np.isfinite(ratio) and ratio >= 0.80)
    row["train_cagr_ge_4pct"] = bool(np.isfinite(train_cagr) and train_cagr >= 0.04)
    row["validation_cagr_ge_3pct"] = bool(np.isfinite(validation_cagr) and validation_cagr >= 0.03)
    row["exposure_active_enough"] = bool(np.isfinite(avg_abs) and avg_abs >= 0.15)
    row["verified_sharpe_robust"] = bool(
        row["train_sharpe_gt_1"]
        and row["validation_sharpe_gt_1"]
        and row["validation_sharpe_ge_80pct_train"]
        and row["train_cagr_ge_4pct"]
        and row["validation_cagr_ge_3pct"]
        and int(row.get("train_years_positive", 0) or 0) >= 10
        and int(row.get("validation_years_positive", 0) or 0) >= 9
        and row["exposure_active_enough"]
        and not bool(row.get("locked_opened"))
    )


def _calmar_rejection_reason(row: dict[str, object]) -> str:
    if int(row.get("train_years_total", 0) or 0) == 0:
        return "no_train_years"
    train_calmar = _finite_float(row.get("train_calmar"), default=-1.0)
    if not np.isfinite(train_calmar) or train_calmar <= 1.0:
        return "train_calmar_below_or_equal_1"
    return ""


def _sharpe_rejection_reason(row: dict[str, object]) -> str:
    if int(row.get("train_years_total", 0) or 0) == 0:
        return "no_train_years"
    train_sharpe = _finite_float(row.get("train_sharpe"), default=-1.0)
    if not np.isfinite(train_sharpe) or train_sharpe <= 1.0:
        return "train_sharpe_below_or_equal_1"
    if _finite_float(row.get("train_cagr"), default=-1.0) < 0.04:
        return "train_cagr_below_4pct"
    if int(row.get("train_years_positive", 0) or 0) < 10:
        return "train_positive_years_below_10"
    if _finite_float(row.get("average_abs_exposure"), default=0.0) < 0.15:
        return "average_abs_exposure_below_15pct"
    return ""


def _weekly_calmar_score(row: dict[str, object]) -> float:
    train_calmar = _finite_float(row.get("train_calmar"), default=-1_000_000.0)
    train_cagr = _finite_float(row.get("train_cagr"), default=-1.0)
    train_mdd = _finite_float(row.get("train_mdd"), default=-1.0)
    train_min = _finite_float(row.get("train_min_year_return"), default=-1.0)
    feature_count = int(row.get("feature_count", 0) or 0)
    calmar = min(train_calmar, 1_000_000.0) if np.isfinite(train_calmar) else -1_000_000.0
    return float(
        calmar * 1_000_000.0
        + train_cagr * 100_000.0
        + train_min * 10_000.0
        - abs(train_mdd) * 1_000.0
        - max(0, feature_count - 5) * 10.0
    )


def _weekly_sharpe_score(row: dict[str, object]) -> float:
    train_sharpe = _finite_float(row.get("train_sharpe"), default=-1_000_000.0)
    train_cagr = _finite_float(row.get("train_cagr"), default=-1.0)
    train_mdd = _finite_float(row.get("train_mdd"), default=-1.0)
    train_min = _finite_float(row.get("train_min_year_return"), default=-1.0)
    feature_count = int(row.get("feature_count", 0) or 0)
    sharpe = min(train_sharpe, 1_000_000.0) if np.isfinite(train_sharpe) else -1_000_000.0
    return float(
        sharpe * 1_000_000.0
        + train_cagr * 100_000.0
        + train_min * 10_000.0
        - abs(train_mdd) * 1_000.0
        - max(0, feature_count - 5) * 10.0
    )


def _finite_float(value: object, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _weekly_score(row: dict[str, object]) -> float:
    train_down_total = max(1, int(row.get("train_down_years_total", 0) or 0))
    train_years_total = max(1, int(row.get("train_years_total", 0) or 0))
    down_passed = int(row.get("train_down_years_ge_5pct", 0) or 0)
    years_positive = int(row.get("train_years_positive", 0) or 0)
    train_down_min = float(row.get("train_down_min_return", -1.0) if pd.notna(row.get("train_down_min_return", np.nan)) else -1.0)
    train_min = float(row.get("train_min_year_return", -1.0) if pd.notna(row.get("train_min_year_return", np.nan)) else -1.0)
    train_cagr = float(row.get("train_cagr", 0.0) or 0.0)
    train_mdd = float(row.get("train_mdd", 0.0) or 0.0)
    turnover = float(row.get("exposure_turnover", 0.0) or 0.0)
    avg_abs = float(row.get("average_abs_exposure", 0.0) or 0.0)
    feature_count = int(row.get("feature_count", 0) or 0)
    down_bonus = 50_000.0 if down_passed == train_down_total and train_down_min >= MIN_DOWN_YEAR_RETURN else 0.0
    positive_bonus = 25_000.0 if years_positive == train_years_total and train_min > 0.0 else 0.0
    mdd_penalty = max(0.0, abs(train_mdd) - 0.30) * 700.0
    return float(
        down_bonus
        + positive_bonus
        + down_passed * 600.0
        + years_positive * 160.0
        + train_down_min * 6_000.0
        + train_min * 3_000.0
        + train_cagr * 750.0
        - mdd_penalty
        - turnover * 180.0
        - max(0.0, avg_abs - 0.80) * 100.0
        - max(0, feature_count - 5) * 12.0
    )


def _verified(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    if "verified_sharpe_robust" in rows:
        sharpe_mask = rows["verified_sharpe_robust"].astype(bool)
        if sharpe_mask.any():
            return rows.loc[sharpe_mask].sort_values("weekly_multi_asset_score", ascending=False).copy()
    if "verified_calmar_similarity" in rows:
        calmar_mask = rows["verified_calmar_similarity"].astype(bool)
        if calmar_mask.any():
            return rows.loc[calmar_mask].sort_values("weekly_multi_asset_score", ascending=False).copy()
    required = {
        "accepted",
        "validation_years_positive",
        "validation_years_total",
        "validation_down_years_ge_5pct",
        "validation_down_years_total",
        "locked_opened",
    }
    if not required.issubset(rows.columns):
        return rows.iloc[0:0].copy()
    mask = (
        rows["accepted"].astype(bool)
        & (pd.to_numeric(rows["validation_years_total"], errors="coerce") > 0)
        & (pd.to_numeric(rows["validation_down_years_total"], errors="coerce") > 0)
        & (
            pd.to_numeric(rows["validation_years_positive"], errors="coerce")
            == pd.to_numeric(rows["validation_years_total"], errors="coerce")
        )
        & (
            pd.to_numeric(rows["validation_down_years_ge_5pct"], errors="coerce")
            == pd.to_numeric(rows["validation_down_years_total"], errors="coerce")
        )
        & ~rows["locked_opened"].astype(bool)
    )
    return rows.loc[mask].sort_values("weekly_multi_asset_score", ascending=False).copy()


def _method_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "method" not in rows:
        return pd.DataFrame()
    out: list[dict[str, object]] = []
    for method, group in rows.groupby("method", sort=True):
        group = group.sort_values("weekly_multi_asset_score", ascending=False)
        verified = _verified(group)
        out.append(
            {
                "method": method,
                "rows": int(len(group)),
                "accepted": int(group.get("accepted", pd.Series(dtype=bool)).astype(bool).sum()),
                "verified_train_validation_5pct": int(len(verified)),
                "best_candidate": str(group.iloc[0]["candidate_id"]) if not group.empty else None,
                "best_score": float(group.iloc[0].get("weekly_multi_asset_score", np.nan)) if not group.empty else None,
                "best_train_min_year_return": float(group.iloc[0].get("train_min_year_return", np.nan)) if not group.empty else None,
                "best_validation_min_year_return": float(group.iloc[0].get("validation_min_year_return", np.nan)) if not group.empty else None,
                "best_train_down_min_return": float(group.iloc[0].get("train_down_min_return", np.nan)) if not group.empty else None,
                "best_validation_down_min_return": float(group.iloc[0].get("validation_down_min_return", np.nan)) if not group.empty else None,
            }
        )
    return pd.DataFrame(out)


def _summary(rows: list[dict[str, object]], *, method: str, stage: int | None = None) -> dict[str, object]:
    best = rows[0] if rows else {}
    return {
        "method": method,
        "stage": stage,
        "candidates_evaluated": int(len(rows)),
        "accepted": int(sum(bool(row.get("accepted")) for row in rows)),
        "verified_train_validation_5pct": int(sum(bool(row.get("verified_train_validation_5pct")) for row in rows)),
        "best_candidate": best.get("candidate_id"),
        "best_assets": best.get("assets"),
        "best_asset_selector": best.get("asset_selector"),
        "best_train_min_year_return": best.get("train_min_year_return"),
        "best_validation_min_year_return": best.get("validation_min_year_return"),
        "best_train_down_min_return": best.get("train_down_min_return"),
        "best_validation_down_min_return": best.get("validation_down_min_return"),
        "locked_opened": False,
        "score_mode": "train_only_weekly_sp500_down_5pct",
        "validation_role": "report_only",
    }


def _empty_summary() -> dict[str, object]:
    return {
        "rows": 0,
        "candidates_evaluated": 0,
        "accepted": 0,
        "verified_train_validation_5pct": 0,
        "unique_verified_train_validation_5pct": 0,
        "locked_opened": False,
        "score_mode": "train_only_weekly_sp500_down_5pct",
        "validation_role": "report_only",
        "tradable_assets": list(TRADABLE_ASSETS),
        "methods": list(WEEKLY_METHODS),
    }


def _method_offset(method: str) -> int:
    return {
        "random_broad": 11_000,
        "beam": 23_000,
        "genetic": 37_000,
        "machine_learning": 43_000,
        "bayesian_like": 41_000,
        "bandit": 53_000,
    }.get(method, 0)


def _ml_feature_names(examples: pd.DataFrame) -> list[str]:
    skip = {
        "decision_date",
        "target_week_end",
        "target_year",
        "target_month",
        "target_week",
        "spy_return_next_week",
        "locked_period",
        "sp500_down_year",
    }
    skip.update({column for column in examples.columns if column.endswith("_return_next_week")})
    names = []
    usable = _without_locked_weekly(examples)
    train = usable.loc[_period_labels_weekly(usable) == "train"]
    for column in usable.columns:
        if column in skip:
            continue
        series = pd.to_numeric(train[column], errors="coerce")
        if series.notna().sum() >= max(52, int(len(train) * 0.20)) and series.nunique(dropna=True) >= 2:
            names.append(column)
    return sorted(names)


def _fit_predict_ml(
    candidate: WeeklyMachineLearningCandidate,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
) -> np.ndarray:
    if candidate.model == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=float(candidate.alpha), random_state=int(candidate.random_seed))
    elif candidate.model == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=int(candidate.n_estimators),
            max_depth=int(candidate.max_depth),
            min_samples_leaf=8,
            random_state=int(candidate.random_seed),
            n_jobs=1,
        )
    elif candidate.model == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=80,
            max_leaf_nodes=max(4, int(candidate.max_depth) * 4),
            learning_rate=float(candidate.learning_rate),
            l2_regularization=float(candidate.alpha),
            random_state=int(candidate.random_seed),
        )
    else:
        raise ValueError(f"unknown weekly ML model: {candidate.model}")
    model.fit(np.asarray(x_train, dtype=float), np.asarray(y_train, dtype=float))
    return np.asarray(model.predict(np.asarray(x_predict, dtype=float)), dtype=float).reshape(-1)


def _ml_candidate_id(candidate: WeeklyMachineLearningCandidate) -> str:
    payload = json.dumps(
        {
            "features": sorted(candidate.features),
            "assets": sorted(candidate.assets),
            "selector": candidate.selector,
            "model": candidate.model,
            "alpha": round(float(candidate.alpha), 6),
            "n_estimators": int(candidate.n_estimators),
            "max_depth": int(candidate.max_depth),
            "learning_rate": round(float(candidate.learning_rate), 6),
            "scale": round(float(candidate.scale), 6),
            "intercept": round(float(candidate.intercept), 6),
            "random_seed": int(candidate.random_seed),
        },
        sort_keys=True,
    )
    return "weekly_ml_asset_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
