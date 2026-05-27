from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_lab.annual_prediction import _build_daily_feature_frame, _political_features


TRAIN_START = pd.Timestamp("1980-01-01")
TRAIN_END = pd.Timestamp("2007-12-31")
VALIDATION_START = pd.Timestamp("2008-01-01")
VALIDATION_END = pd.Timestamp("2019-12-31")
LOCKED_START = pd.Timestamp("2020-01-01")
MIN_TRAIN_YEAR_RETURN = 0.10


@dataclass(frozen=True)
class MonthlyRiskCandidate:
    specs: tuple[str, ...]
    intercept: float = 0.0
    scale: float = 1.0
    smoothing: float = 0.0


@dataclass(frozen=True)
class MonthlyRiskSearchConfig:
    stage: int
    total_stages: int = 64
    seed_pool: int = 600
    beam_width: int = 40
    generations: int = 6
    mutations_per_parent: int = 12
    max_features: int = 6
    random_seed: int = 412_000
    top_rows_per_stage: int = 250
    score_mode: str = "train_only_weekly_sp500_down_5pct"


def build_monthly_examples(
    daily: pd.DataFrame,
    *,
    start_year: int = 1980,
    end_year: int | None = None,
) -> pd.DataFrame:
    data = _normalize_daily(daily)
    end_year = end_year or int(data.index.max().year)
    features = _build_daily_feature_frame(data)
    month_ends = data.resample("ME").last().dropna(subset=["close"])
    rows: list[dict[str, object]] = []
    for index in range(len(month_ends) - 1):
        decision_date = month_ends.index[index]
        target_end = month_ends.index[index + 1]
        target_year = int(target_end.year)
        if target_year < start_year or target_year > end_year:
            continue
        decision_close = float(month_ends["close"].iloc[index])
        target_close = float(month_ends["close"].iloc[index + 1])
        if not np.isfinite(decision_close) or decision_close <= 0.0:
            continue
        feature_date = data.index[data.index <= decision_date][-1]
        feature_row = features.loc[feature_date]
        row: dict[str, object] = {
            "decision_date": decision_date,
            "target_month_end": target_end,
            "target_year": target_year,
            "target_month": int(target_end.month),
            "spy_return_next_month": target_close / decision_close - 1.0,
            "month_number": float(target_end.month),
            "quarter_number": float(((target_end.month - 1) // 3) + 1),
            "locked_period": bool(target_end >= LOCKED_START),
            **_political_features(target_year),
        }
        for name, value in feature_row.items():
            row[name] = float(value) if pd.notna(value) else np.nan
        rows.append(row)
    examples = pd.DataFrame(rows)
    if examples.empty:
        return examples
    examples["decision_date"] = pd.to_datetime(examples["decision_date"])
    examples["target_month_end"] = pd.to_datetime(examples["target_month_end"])
    examples["target_year"] = examples["target_year"].astype(int)
    examples["target_month"] = examples["target_month"].astype(int)
    return examples.replace([np.inf, -np.inf], np.nan)


def evaluate_monthly_candidate(
    examples: pd.DataFrame,
    candidate: MonthlyRiskCandidate,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    usable = _without_locked(examples)
    exposures = _candidate_exposure(usable, candidate)
    returns = pd.to_numeric(usable["spy_return_next_month"], errors="coerce").fillna(0.0).to_numpy()
    strategy_returns = exposures * returns
    positions = pd.DataFrame(
        {
            "candidate_id": _candidate_id(candidate),
            "decision_date": pd.to_datetime(usable["decision_date"]).to_numpy(),
            "target_month_end": pd.to_datetime(usable["target_month_end"]).to_numpy()
            if "target_month_end" in usable
            else pd.to_datetime(usable["decision_date"]).to_numpy(),
            "target_year": usable["target_year"].astype(int).to_numpy(),
            "target_month": usable["target_month"].astype(int).to_numpy(),
            "period": _period_labels(usable),
            "spy_return": returns,
            "exposure": exposures,
            "strategy_return": strategy_returns,
        }
    )
    year_by_year = _year_by_year(positions)
    train_metrics = _period_return_metrics(positions.loc[positions["period"] == "train", "strategy_return"])
    validation_metrics = _period_return_metrics(positions.loc[positions["period"] == "validation", "strategy_return"])
    train_years = year_by_year[year_by_year["period"] == "train"]
    validation_years = year_by_year[year_by_year["period"] == "validation"]
    train_min_year = float(train_years["strategy_return"].min()) if not train_years.empty else np.nan
    validation_min_year = float(validation_years["strategy_return"].min()) if not validation_years.empty else np.nan
    train_years_ge = int((train_years["strategy_return"] >= MIN_TRAIN_YEAR_RETURN).sum()) if not train_years.empty else 0
    validation_years_ge = int((validation_years["strategy_return"] >= MIN_TRAIN_YEAR_RETURN).sum()) if not validation_years.empty else 0
    candidate_id = _candidate_id(candidate)
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "specs": ";".join(candidate.specs),
        "rules": _rules_text(candidate),
        "features": ",".join(_candidate_features(candidate)),
        "feature_count": len(_candidate_features(candidate)),
        "monthly_exposure_formula": _formula_text(candidate),
        "intercept": float(candidate.intercept),
        "scale": float(candidate.scale),
        "smoothing": float(candidate.smoothing),
        "train_years_ge_10pct": train_years_ge,
        "train_years_total": int(len(train_years)),
        "validation_years_ge_10pct": validation_years_ge,
        "validation_years_total": int(len(validation_years)),
        "train_min_year_return": train_min_year,
        "validation_min_year_return": validation_min_year,
        "train_cagr": train_metrics["cagr"],
        "validation_cagr": validation_metrics["cagr"],
        "train_mdd": train_metrics["mdd"],
        "validation_mdd": validation_metrics["mdd"],
        "train_calmar": train_metrics["calmar"],
        "validation_calmar": validation_metrics["calmar"],
        "average_exposure": float(np.mean(exposures)) if len(exposures) else 0.0,
        "average_abs_exposure": float(np.mean(np.abs(exposures))) if len(exposures) else 0.0,
        "min_exposure": float(np.min(exposures)) if len(exposures) else 0.0,
        "max_exposure": float(np.max(exposures)) if len(exposures) else 0.0,
        "months_long": int(np.sum(exposures > 0.05)),
        "months_cash_like": int(np.sum(np.abs(exposures) <= 0.05)),
        "months_short": int(np.sum(exposures < -0.05)),
        "worst_month": float(np.min(strategy_returns)) if len(strategy_returns) else 0.0,
        "exposure_turnover": float(np.mean(np.abs(np.diff(exposures)))) if len(exposures) > 1 else 0.0,
        "locked_opened": False,
        "locked_months": 0,
        "accepted": False,
        "rejection_reason": "",
        "monthly_risk_score": 0.0,
    }
    row["rejection_reason"] = _rejection_reason(row)
    row["accepted"] = row["rejection_reason"] == ""
    row["monthly_risk_score"] = _monthly_risk_score(row)
    return row, positions, year_by_year


def run_monthly_risk_search(
    examples: pd.DataFrame,
    config: MonthlyRiskSearchConfig,
) -> list[dict[str, object]]:
    catalog = _build_spec_catalog(examples)
    if not catalog:
        return []
    rng = np.random.default_rng(config.random_seed + config.stage)
    rows: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, ...], float, float, float]] = set()
    seeds = _seed_candidates(catalog, config=config, rng=rng)
    seed_rows = _evaluate_unique(examples, seeds, seen)
    rows.extend(seed_rows)
    beam = _select_beam(seed_rows, config.beam_width)
    for _ in range(config.generations):
        children: list[MonthlyRiskCandidate] = []
        for parent in beam:
            candidate = _candidate_from_row(parent)
            for _ in range(config.mutations_per_parent):
                children.append(_mutate_candidate(candidate, catalog, config=config, rng=rng))
        child_rows = _evaluate_unique(examples, children, seen)
        rows.extend(child_rows)
        beam = _select_beam([*beam, *child_rows], config.beam_width)
    sorted_rows = sorted(rows, key=lambda row: float(row["monthly_risk_score"]), reverse=True)
    if config.top_rows_per_stage > 0:
        accepted = [row for row in sorted_rows if bool(row.get("accepted"))]
        top = sorted_rows[: config.top_rows_per_stage]
        by_id = {str(row["candidate_id"]): row for row in [*top, *accepted]}
        sorted_rows = sorted(by_id.values(), key=lambda row: float(row["monthly_risk_score"]), reverse=True)
    return sorted_rows


def write_monthly_risk_outputs(
    rows: list[dict[str, object]],
    examples: pd.DataFrame,
    output_dir: str | Path,
    *,
    stage: int | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.DataFrame(rows)
    leaderboard.to_csv(output / "monthly_risk_leaderboard.csv", index=False)
    if stage is not None:
        leaderboard.to_csv(output / f"monthly_risk_leaderboard_stage_{stage}.csv", index=False)
    with (output / "monthly_risk_candidates.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
    summary = _summary(rows, stage=stage)
    (output / "monthly_risk_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        best = _candidate_from_row(rows[0])
        _, positions, year_by_year = evaluate_monthly_candidate(examples, best)
    else:
        positions = pd.DataFrame()
        year_by_year = pd.DataFrame()
    positions.to_csv(output / "monthly_risk_monthly_positions.csv", index=False)
    year_by_year.to_csv(output / "monthly_risk_year_by_year.csv", index=False)


def merge_monthly_risk_leaderboards(paths: list[str | Path], output_dir: str | Path) -> dict[str, object]:
    frames = [pd.read_csv(path) for path in paths if Path(path).exists()]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not frames:
        empty = pd.DataFrame()
        empty.to_csv(output / "monthly_risk_leaderboard.csv", index=False)
        empty.to_csv(output / "monthly_risk_train_validation_10pct.csv", index=False)
        summary = {
            "rows": 0,
            "accepted": 0,
            "verified_train_validation_10pct": 0,
            "unique_verified_train_validation_10pct": 0,
            "locked_opened": False,
            "score_mode": "train_only_monthly_10pct",
            "validation_role": "report_only",
        }
        (output / "monthly_risk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    merged = pd.concat(frames, ignore_index=True)
    if "candidate_id" in merged:
        merged = merged.sort_values("monthly_risk_score", ascending=False).drop_duplicates("candidate_id", keep="first")
    merged = merged.sort_values("monthly_risk_score", ascending=False)
    merged.to_csv(output / "monthly_risk_leaderboard.csv", index=False)
    verified = _verified_train_validation_10pct(merged)
    verified.to_csv(output / "monthly_risk_train_validation_10pct.csv", index=False)
    summary = {
        "rows": int(len(merged)),
        "accepted": int(merged.get("accepted", pd.Series(dtype=bool)).astype(bool).sum()),
        "verified_train_validation_10pct": int(len(verified)),
        "unique_verified_train_validation_10pct": int(verified["candidate_id"].nunique()) if "candidate_id" in verified else 0,
        "best_candidate": str(merged.iloc[0]["candidate_id"]) if not merged.empty else None,
        "best_train_min_year_return": float(merged.iloc[0].get("train_min_year_return", np.nan)) if not merged.empty else None,
        "best_validation_min_year_return": float(merged.iloc[0].get("validation_min_year_return", np.nan)) if not merged.empty else None,
        "best_verified_candidate": str(verified.iloc[0]["candidate_id"]) if not verified.empty else None,
        "locked_opened": False,
        "score_mode": "train_only_monthly_10pct",
        "validation_role": "report_only",
    }
    (output / "monthly_risk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _verified_train_validation_10pct(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    required = {
        "accepted",
        "train_years_ge_10pct",
        "train_years_total",
        "validation_years_ge_10pct",
        "validation_years_total",
        "locked_opened",
    }
    if not required.issubset(rows.columns):
        return rows.iloc[0:0].copy()
    mask = (
        rows["accepted"].astype(bool)
        & (pd.to_numeric(rows["train_years_total"], errors="coerce") > 0)
        & (pd.to_numeric(rows["validation_years_total"], errors="coerce") > 0)
        & (
            pd.to_numeric(rows["train_years_ge_10pct"], errors="coerce")
            == pd.to_numeric(rows["train_years_total"], errors="coerce")
        )
        & (
            pd.to_numeric(rows["validation_years_ge_10pct"], errors="coerce")
            == pd.to_numeric(rows["validation_years_total"], errors="coerce")
        )
        & ~rows["locked_opened"].astype(bool)
    )
    return rows.loc[mask].sort_values("monthly_risk_score", ascending=False).copy()


def _normalize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily.copy()
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.set_index("timestamp")
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    if "close" not in data.columns:
        raise ValueError("monthly risk data needs a close column")
    return data


def _without_locked(examples: pd.DataFrame) -> pd.DataFrame:
    target_dates = _target_dates(examples)
    return examples.loc[target_dates < LOCKED_START].copy()


def _target_dates(examples: pd.DataFrame) -> pd.Series:
    if "target_month_end" in examples:
        return pd.to_datetime(examples["target_month_end"])
    years = examples["target_year"].astype(int).astype(str)
    months = examples["target_month"].astype(int).astype(str).str.zfill(2)
    return pd.to_datetime(years + "-" + months + "-01") + pd.offsets.MonthEnd(0)


def _period_labels(examples: pd.DataFrame) -> np.ndarray:
    dates = _target_dates(examples)
    labels = np.where(dates <= TRAIN_END, "train", np.where(dates <= VALIDATION_END, "validation", "locked"))
    return labels.astype(object)


def _candidate_exposure(examples: pd.DataFrame, candidate: MonthlyRiskCandidate) -> np.ndarray:
    if examples.empty:
        return np.array([], dtype=float)
    score = np.zeros(len(examples), dtype=float)
    total_weight = 0.0
    for spec in candidate.specs:
        signal, weight = _spec_signal(examples, spec)
        score += signal * weight
        total_weight += abs(weight)
    if total_weight > 0.0:
        score = score / total_weight
    raw = np.clip(candidate.intercept + candidate.scale * score, -1.0, 1.0)
    smoothing = min(max(float(candidate.smoothing), 0.0), 0.95)
    if smoothing <= 0.0 or len(raw) <= 1:
        return raw.astype(float)
    smoothed = np.empty_like(raw)
    smoothed[0] = raw[0]
    for index in range(1, len(raw)):
        smoothed[index] = smoothing * smoothed[index - 1] + (1.0 - smoothing) * raw[index]
    return np.clip(smoothed, -1.0, 1.0).astype(float)


def _spec_signal(examples: pd.DataFrame, spec: str) -> tuple[np.ndarray, float]:
    parts = spec.split("|")
    feature = parts[0]
    kind = parts[1]
    values = pd.to_numeric(examples.get(feature, pd.Series(index=examples.index, dtype=float)), errors="coerce")
    values = values.fillna(values.median()).fillna(0.0)
    direction = float(parts[-2])
    weight = float(parts[-1])
    if kind == "range":
        low = float(parts[2])
        high = float(parts[3])
        active = (values >= low) & (values <= high)
    else:
        threshold = float(parts[2])
        active = values >= threshold
    signal = np.where(active.to_numpy(), 1.0, -1.0) * direction
    return signal.astype(float), weight


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
                "period": period,
                "year": int(year),
                "strategy_return": strategy_return,
                "spy_return": spy_return,
                "excess_return": strategy_return - spy_return,
                "average_exposure": float(group["exposure"].mean()),
                "min_exposure": float(group["exposure"].min()),
                "max_exposure": float(group["exposure"].max()),
                "months": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _period_return_metrics(returns: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(values) == 0:
        return {"cagr": 0.0, "mdd": 0.0, "calmar": 0.0}
    equity = np.cumprod(1.0 + values)
    years = max(len(values) / 12.0, 1e-9)
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0.0 else -1.0
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    mdd = float(drawdown.min())
    calmar = float(cagr / abs(mdd)) if mdd < 0.0 else float("inf") if cagr > 0 else 0.0
    return {"cagr": cagr, "mdd": mdd, "calmar": calmar}


def _rejection_reason(row: dict[str, object]) -> str:
    total = int(row.get("train_years_total", 0) or 0)
    passed = int(row.get("train_years_ge_10pct", 0) or 0)
    if total == 0:
        return "no_train_years"
    if passed < total:
        return "train_year_below_10pct"
    return ""


def _monthly_risk_score(row: dict[str, object]) -> float:
    train_min = float(row.get("train_min_year_return", -1.0) or -1.0)
    train_cagr = float(row.get("train_cagr", 0.0) or 0.0)
    train_mdd = float(row.get("train_mdd", 0.0) or 0.0)
    passed = int(row.get("train_years_ge_10pct", 0) or 0)
    total = max(1, int(row.get("train_years_total", 0) or 0))
    turnover = float(row.get("exposure_turnover", 0.0) or 0.0)
    avg_abs = float(row.get("average_abs_exposure", 0.0) or 0.0)
    feature_count = int(row.get("feature_count", 0) or 0)
    perfect_bonus = 10_000.0 if passed == total and train_min >= MIN_TRAIN_YEAR_RETURN else 0.0
    mdd_penalty = max(0.0, abs(train_mdd) - 0.25) * 600.0
    return float(
        perfect_bonus
        + passed * 120.0
        + train_min * 2_000.0
        + train_cagr * 500.0
        - mdd_penalty
        - turnover * 120.0
        - max(0.0, avg_abs - 0.75) * 80.0
        - max(0, feature_count - 3) * 15.0
    )


def _build_spec_catalog(examples: pd.DataFrame) -> list[str]:
    usable = _without_locked(examples)
    train = usable.loc[_period_labels(usable) == "train"]
    if train.empty:
        return []
    skip = {
        "decision_date",
        "target_month_end",
        "target_year",
        "target_month",
        "spy_return_next_month",
        "locked_period",
    }
    catalog: list[str] = []
    for column in usable.columns:
        if column in skip:
            continue
        series = pd.to_numeric(train[column], errors="coerce")
        if series.notna().sum() < max(12, int(len(train) * 0.30)):
            continue
        if series.nunique(dropna=True) < 2:
            continue
        quantiles = series.quantile([0.2, 0.35, 0.5, 0.65, 0.8]).dropna().drop_duplicates()
        for threshold in quantiles:
            for direction in (-1, 1):
                for weight in (1, 2):
                    catalog.append(f"{column}|threshold|{float(threshold):.10g}|{direction}|{weight}")
        low_high = series.quantile([0.25, 0.75]).dropna()
        if len(low_high) == 2 and float(low_high.iloc[0]) < float(low_high.iloc[1]):
            for direction in (-1, 1):
                catalog.append(
                    f"{column}|range|{float(low_high.iloc[0]):.10g}|{float(low_high.iloc[1]):.10g}|{direction}|1"
                )
    return sorted(set(catalog))


def _seed_candidates(
    catalog: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
) -> list[MonthlyRiskCandidate]:
    stage_catalog = [spec for index, spec in enumerate(catalog) if index % config.total_stages == config.stage]
    if not stage_catalog:
        stage_catalog = catalog
    candidates: list[MonthlyRiskCandidate] = []
    exposures = (-0.8, -0.5, -0.25, 0.0, 0.25, 0.5, 0.8)
    scales = (0.25, 0.5, 0.75, 1.0)
    for spec in stage_catalog[: config.seed_pool]:
        for scale in scales:
            candidates.append(MonthlyRiskCandidate((spec,), intercept=0.0, scale=scale))
        candidates.append(MonthlyRiskCandidate((spec,), intercept=float(rng.choice(exposures)), scale=0.25))
    while len(candidates) < config.seed_pool:
        size = int(rng.integers(1, min(config.max_features, 4) + 1))
        specs = tuple(sorted(rng.choice(stage_catalog, size=size, replace=False).tolist()))
        candidates.append(
            MonthlyRiskCandidate(
                specs,
                intercept=float(rng.uniform(-0.35, 0.35)),
                scale=float(rng.uniform(0.25, 1.0)),
                smoothing=float(rng.choice([0.0, 0.15, 0.30])),
            )
        )
    return candidates[: config.seed_pool]


def _evaluate_unique(
    examples: pd.DataFrame,
    candidates: list[MonthlyRiskCandidate],
    seen: set[tuple[tuple[str, ...], float, float, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        key = (
            tuple(sorted(candidate.specs)),
            round(float(candidate.intercept), 4),
            round(float(candidate.scale), 4),
            round(float(candidate.smoothing), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        row, _, _ = evaluate_monthly_candidate(examples, candidate)
        rows.append(row)
    return rows


def _select_beam(rows: list[dict[str, object]], size: int) -> list[dict[str, object]]:
    by_id = {str(row["candidate_id"]): row for row in rows}
    return sorted(by_id.values(), key=lambda row: float(row["monthly_risk_score"]), reverse=True)[:size]


def _mutate_candidate(
    candidate: MonthlyRiskCandidate,
    catalog: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
) -> MonthlyRiskCandidate:
    specs = list(candidate.specs)
    action = str(rng.choice(["add", "replace", "drop", "tweak"]))
    if action == "add" and len(specs) < config.max_features:
        specs.append(str(rng.choice(catalog)))
    elif action == "replace" and specs:
        specs[int(rng.integers(0, len(specs)))] = str(rng.choice(catalog))
    elif action == "drop" and len(specs) > 1:
        specs.pop(int(rng.integers(0, len(specs))))
    else:
        specs = _tweak_spec_weight(specs, rng)
    intercept = float(np.clip(candidate.intercept + rng.normal(0.0, 0.12), -0.8, 0.8))
    scale = float(np.clip(candidate.scale + rng.normal(0.0, 0.18), 0.05, 1.50))
    smoothing = float(np.clip(candidate.smoothing + rng.normal(0.0, 0.10), 0.0, 0.70))
    return MonthlyRiskCandidate(tuple(sorted(set(specs))), intercept=intercept, scale=scale, smoothing=smoothing)


def _tweak_spec_weight(specs: list[str], rng: np.random.Generator) -> list[str]:
    if not specs:
        return specs
    index = int(rng.integers(0, len(specs)))
    parts = specs[index].split("|")
    parts[-1] = str(int(rng.choice([1, 2, 3])))
    specs[index] = "|".join(parts)
    return specs


def _candidate_from_row(row: dict[str, object] | pd.Series) -> MonthlyRiskCandidate:
    specs = tuple(part for part in str(row.get("specs", "")).split(";") if part)
    return MonthlyRiskCandidate(
        specs,
        intercept=float(row.get("intercept", 0.0) or 0.0),
        scale=float(row.get("scale", 1.0) or 1.0),
        smoothing=float(row.get("smoothing", 0.0) or 0.0),
    )


def _candidate_id(candidate: MonthlyRiskCandidate) -> str:
    payload = json.dumps(
        {
            "specs": sorted(candidate.specs),
            "intercept": round(float(candidate.intercept), 6),
            "scale": round(float(candidate.scale), 6),
            "smoothing": round(float(candidate.smoothing), 6),
        },
        sort_keys=True,
    )
    return "monthly_risk_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _candidate_features(candidate: MonthlyRiskCandidate) -> list[str]:
    return sorted({spec.split("|", 1)[0] for spec in candidate.specs})


def _rules_text(candidate: MonthlyRiskCandidate) -> str:
    pieces = []
    for spec in candidate.specs:
        parts = spec.split("|")
        if parts[1] == "range":
            pieces.append(f"{parts[0]} between {parts[2]} and {parts[3]} direction {parts[-2]} weight {parts[-1]}")
        else:
            pieces.append(f"{parts[0]} >= {parts[2]} direction {parts[-2]} weight {parts[-1]}")
    return " ; ".join(pieces)


def _formula_text(candidate: MonthlyRiskCandidate) -> str:
    return (
        "exposure = clip("
        f"{candidate.intercept:.4f} + {candidate.scale:.4f} * weighted_rule_vote"
        ", -1, 1)"
    )


def _summary(rows: list[dict[str, object]], *, stage: int | None = None) -> dict[str, object]:
    best = rows[0] if rows else {}
    return {
        "stage": stage,
        "candidates_evaluated": int(len(rows)),
        "accepted": int(sum(bool(row.get("accepted")) for row in rows)),
        "best_candidate": best.get("candidate_id"),
        "best_train_min_year_return": best.get("train_min_year_return"),
        "best_validation_min_year_return": best.get("validation_min_year_return"),
        "best_train_cagr": best.get("train_cagr"),
        "best_validation_cagr": best.get("validation_cagr"),
        "locked_opened": False,
    }


def _json_safe(row: dict[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, (np.integer,)):
            safe[key] = int(value)
        elif isinstance(value, (np.floating,)):
            safe[key] = float(value)
        elif pd.isna(value) if not isinstance(value, (str, bool, list, tuple, dict)) else False:
            safe[key] = None
        else:
            safe[key] = value
    return safe
