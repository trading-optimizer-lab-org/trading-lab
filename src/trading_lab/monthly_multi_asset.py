from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_lab.monthly_risk import (
    LOCKED_START,
    MIN_TRAIN_YEAR_RETURN,
    MonthlyRiskCandidate,
    MonthlyRiskSearchConfig,
    _build_spec_catalog,
    _candidate_exposure,
    _candidate_features,
    _json_safe,
    _monthly_risk_score,
    _normalize_daily,
    _period_labels,
    _period_return_metrics,
    _rejection_reason,
    _rules_text,
    _target_dates,
    _without_locked,
    build_monthly_examples,
)


TRADABLE_ASSETS = (
    "SPY",
    "QQQ",
    "IWM",
    "RSP",
    "DIA",
    "EFA",
    "EEM",
    "EWJ",
    "EWG",
    "EWU",
    "FXI",
    "TLT",
    "IEF",
    "HYG",
    "LQD",
    "GLD",
    "SLV",
    "USO",
    "CPER",
    "DBC",
    "UUP",
    "SHY",
    "IEI",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLY",
    "XLP",
    "XLU",
    "XLI",
    "XLB",
    "XLRE",
    "XLC",
)

NON_TRADABLE_INDICATORS = ("^VIX", "^VIX3M", "^VVIX", "^SKEW", "^TNX", "^IRX", "^FVX", "^TYX")
ASSET_SELECTORS = ("momentum_3m", "momentum_6m", "momentum_12m", "low_vol_3m")


@dataclass(frozen=True)
class MonthlyMultiAssetCandidate:
    specs: tuple[str, ...]
    assets: tuple[str, ...]
    selector: str = "momentum_6m"
    intercept: float = 0.0
    scale: float = 1.0
    smoothing: float = 0.0


def available_tradable_assets(daily: pd.DataFrame) -> list[dict[str, object]]:
    data = _normalize_daily(daily)
    rows = [{"asset": "SPY", "first_date": str(data.index.min().date()), "last_date": str(data.index.max().date())}]
    for asset in TRADABLE_ASSETS:
        if asset == "SPY":
            continue
        column = _ratio_column(asset)
        if column not in data:
            continue
        series = pd.to_numeric(data[column], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append(
            {
                "asset": asset,
                "first_date": str(series.index.min().date()),
                "last_date": str(series.index.max().date()),
            }
        )
    return rows


def build_monthly_multi_asset_examples(
    daily: pd.DataFrame,
    *,
    start_year: int = 1994,
    end_year: int | None = None,
) -> pd.DataFrame:
    data = _normalize_daily(daily)
    examples = build_monthly_examples(daily, start_year=start_year, end_year=end_year)
    if examples.empty:
        return examples
    monthly_closes = _monthly_asset_closes(data)
    extra_columns: dict[str, np.ndarray] = {}
    for asset in monthly_closes:
        close = monthly_closes[asset]
        next_return = close.shift(-1) / close - 1.0
        extra_columns[_asset_col(asset, "return_next_month")] = _align_monthly_series(examples, next_return)
        for months in (3, 6, 12):
            momentum = close / close.shift(months) - 1.0
            extra_columns[_asset_col(asset, f"mom_{months}m")] = _align_monthly_series(examples, momentum)
        monthly_return = close.pct_change()
        vol = monthly_return.rolling(3).std() * np.sqrt(12.0)
        extra_columns[_asset_col(asset, "vol_3m")] = _align_monthly_series(examples, vol)
    if extra_columns:
        examples = pd.concat([examples, pd.DataFrame(extra_columns, index=examples.index)], axis=1)
    return examples.replace([np.inf, -np.inf], np.nan)


def evaluate_monthly_multi_asset_candidate(
    examples: pd.DataFrame,
    candidate: MonthlyMultiAssetCandidate,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    usable = _without_locked(examples)
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
    spy_returns = pd.to_numeric(usable["spy_return_next_month"], errors="coerce").fillna(0.0).to_numpy()
    positions = pd.DataFrame(
        {
            "candidate_id": _candidate_id(candidate),
            "decision_date": pd.to_datetime(usable["decision_date"]).to_numpy(),
            "target_month_end": pd.to_datetime(usable["target_month_end"]).to_numpy(),
            "target_year": usable["target_year"].astype(int).to_numpy(),
            "target_month": usable["target_month"].astype(int).to_numpy(),
            "period": _period_labels(usable),
            "traded_asset": selected_assets,
            "asset_return": asset_returns,
            "spy_return": spy_returns,
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
    feature_names = _candidate_features(exposure_candidate)
    row: dict[str, object] = {
        "candidate_id": _candidate_id(candidate),
        "specs": ";".join(candidate.specs),
        "rules": _rules_text(exposure_candidate),
        "features": ",".join(feature_names),
        "feature_count": len(feature_names),
        "assets": ",".join(candidate.assets),
        "asset_selector": candidate.selector,
        "traded_asset_mode": "monthly_rotation",
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
        "unique_assets_used": int(pd.Series(selected_assets).nunique()) if len(selected_assets) else 0,
        "most_used_asset": str(pd.Series(selected_assets).mode().iloc[0]) if len(selected_assets) else "",
        "cash_allowed": True,
        "short_allowed": True,
        "locked_opened": False,
        "locked_months": 0,
        "accepted": False,
        "rejection_reason": "",
        "monthly_multi_asset_score": 0.0,
    }
    row["rejection_reason"] = _rejection_reason(row)
    row["accepted"] = row["rejection_reason"] == ""
    row["monthly_multi_asset_score"] = _multi_asset_score(row)
    return row, positions, year_by_year


def run_monthly_multi_asset_search(
    examples: pd.DataFrame,
    config: MonthlyRiskSearchConfig,
) -> list[dict[str, object]]:
    catalog = _build_multi_asset_spec_catalog(examples)
    assets = _available_assets_from_examples(examples)
    if not catalog or not assets:
        return []
    rng = np.random.default_rng(config.random_seed + config.stage)
    rows: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str, float, float, float]] = set()
    seeds = _seed_candidates(catalog, assets, config=config, rng=rng)
    seed_rows = _evaluate_unique(examples, seeds, seen)
    rows.extend(seed_rows)
    beam = _select_beam(seed_rows, config.beam_width)
    for _ in range(config.generations):
        children: list[MonthlyMultiAssetCandidate] = []
        for parent in beam:
            candidate = _candidate_from_row(parent)
            for _ in range(config.mutations_per_parent):
                children.append(_mutate_candidate(candidate, catalog, assets, config=config, rng=rng))
        child_rows = _evaluate_unique(examples, children, seen)
        rows.extend(child_rows)
        beam = _select_beam([*beam, *child_rows], config.beam_width)
    sorted_rows = sorted(rows, key=lambda row: float(row["monthly_multi_asset_score"]), reverse=True)
    if config.top_rows_per_stage > 0:
        accepted = [row for row in sorted_rows if bool(row.get("accepted"))]
        top = sorted_rows[: config.top_rows_per_stage]
        by_id = {str(row["candidate_id"]): row for row in [*top, *accepted]}
        sorted_rows = sorted(by_id.values(), key=lambda row: float(row["monthly_multi_asset_score"]), reverse=True)
    return sorted_rows


def write_monthly_multi_asset_outputs(
    rows: list[dict[str, object]],
    examples: pd.DataFrame,
    output_dir: str | Path,
    *,
    stage: int | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.DataFrame(rows)
    leaderboard.to_csv(output / "monthly_multi_asset_leaderboard.csv", index=False)
    if stage is not None:
        leaderboard.to_csv(output / f"monthly_multi_asset_leaderboard_stage_{stage}.csv", index=False)
    with (output / "monthly_multi_asset_candidates.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
    summary = _summary(rows, stage=stage)
    (output / "monthly_multi_asset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if rows:
        best = _candidate_from_row(rows[0])
        _, positions, year_by_year = evaluate_monthly_multi_asset_candidate(examples, best)
    else:
        positions = pd.DataFrame()
        year_by_year = pd.DataFrame()
    positions.to_csv(output / "monthly_multi_asset_monthly_positions.csv", index=False)
    year_by_year.to_csv(output / "monthly_multi_asset_year_by_year.csv", index=False)


def merge_monthly_multi_asset_leaderboards(paths: list[str | Path], output_dir: str | Path) -> dict[str, object]:
    frames = [pd.read_csv(path) for path in paths if Path(path).exists()]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not frames:
        empty = pd.DataFrame()
        empty.to_csv(output / "monthly_multi_asset_leaderboard.csv", index=False)
        empty.to_csv(output / "monthly_multi_asset_train_validation_10pct.csv", index=False)
        summary = _empty_summary()
        (output / "monthly_multi_asset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    merged = pd.concat(frames, ignore_index=True)
    if "candidate_id" in merged:
        merged = (
            merged.sort_values("monthly_multi_asset_score", ascending=False)
            .drop_duplicates("candidate_id", keep="first")
        )
    merged = merged.sort_values("monthly_multi_asset_score", ascending=False)
    merged.to_csv(output / "monthly_multi_asset_leaderboard.csv", index=False)
    verified = _verified_train_validation_10pct(merged)
    verified.to_csv(output / "monthly_multi_asset_train_validation_10pct.csv", index=False)
    summary = {
        "rows": int(len(merged)),
        "accepted": int(merged.get("accepted", pd.Series(dtype=bool)).astype(bool).sum()),
        "verified_train_validation_10pct": int(len(verified)),
        "unique_verified_train_validation_10pct": int(verified["candidate_id"].nunique()) if "candidate_id" in verified else 0,
        "best_candidate": str(merged.iloc[0]["candidate_id"]) if not merged.empty else None,
        "best_assets": str(merged.iloc[0].get("assets", "")) if not merged.empty else None,
        "best_asset_selector": str(merged.iloc[0].get("asset_selector", "")) if not merged.empty else None,
        "best_train_min_year_return": float(merged.iloc[0].get("train_min_year_return", np.nan)) if not merged.empty else None,
        "best_validation_min_year_return": float(merged.iloc[0].get("validation_min_year_return", np.nan)) if not merged.empty else None,
        "best_verified_candidate": str(verified.iloc[0]["candidate_id"]) if not verified.empty else None,
        "locked_opened": False,
        "score_mode": "train_only_monthly_multi_asset_10pct",
        "validation_role": "report_only",
        "tradable_assets": list(TRADABLE_ASSETS),
        "non_tradable_indicators": list(NON_TRADABLE_INDICATORS),
    }
    (output / "monthly_multi_asset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _monthly_asset_closes(data: pd.DataFrame) -> dict[str, pd.Series]:
    closes = {"SPY": pd.to_numeric(data["close"], errors="coerce")}
    for asset in TRADABLE_ASSETS:
        if asset == "SPY":
            continue
        ratio_column = _ratio_column(asset)
        if ratio_column not in data:
            continue
        ratio = pd.to_numeric(data[ratio_column], errors="coerce")
        close = closes["SPY"] * ratio
        if close.notna().sum() >= 24:
            closes[asset] = close
    return {asset: close.resample("ME").last() for asset, close in closes.items()}


def _align_monthly_series(examples: pd.DataFrame, series: pd.Series) -> np.ndarray:
    lookup = series.copy()
    lookup.index = pd.to_datetime(lookup.index)
    values = lookup.reindex(pd.to_datetime(examples["decision_date"])).to_numpy(dtype=float)
    return values


def _build_multi_asset_spec_catalog(examples: pd.DataFrame) -> list[str]:
    cleaned = examples.drop(columns=[c for c in examples.columns if c.endswith("_return_next_month")], errors="ignore")
    return _build_spec_catalog(cleaned)


def _available_assets_from_examples(examples: pd.DataFrame) -> list[str]:
    assets = []
    for asset in TRADABLE_ASSETS:
        column = _asset_col(asset, "return_next_month")
        if column in examples and pd.to_numeric(examples[column], errors="coerce").notna().sum() >= 24:
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


def _select_assets(examples: pd.DataFrame, candidate: MonthlyMultiAssetCandidate) -> np.ndarray:
    assets = [asset for asset in candidate.assets if _asset_col(asset, "return_next_month") in examples]
    if not assets:
        assets = ["SPY"]
    score_suffix = {
        "momentum_3m": "mom_3m",
        "momentum_6m": "mom_6m",
        "momentum_12m": "mom_12m",
        "low_vol_3m": "vol_3m",
    }.get(candidate.selector, "mom_6m")
    score_frame = pd.DataFrame(index=examples.index)
    for asset in assets:
        column = _asset_col(asset, score_suffix)
        values = pd.to_numeric(examples.get(column, pd.Series(index=examples.index, dtype=float)), errors="coerce")
        if candidate.selector == "low_vol_3m":
            values = -values
        score_frame[asset] = values
    selected: list[str] = []
    for _, row in score_frame.iterrows():
        valid = row.dropna()
        selected.append(str(valid.idxmax()) if not valid.empty else "SPY")
    return np.array(selected, dtype=object)


def _selected_asset_returns(examples: pd.DataFrame, selected_assets: np.ndarray) -> np.ndarray:
    returns = np.zeros(len(examples), dtype=float)
    for index, asset in enumerate(selected_assets):
        column = _asset_col(str(asset), "return_next_month")
        if column in examples:
            value = pd.to_numeric(pd.Series([examples.iloc[index][column]]), errors="coerce").iloc[0]
            returns[index] = float(value) if pd.notna(value) else 0.0
    return returns


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
                "dominant_asset": str(group["traded_asset"].mode().iloc[0]),
                "unique_assets_used": int(group["traded_asset"].nunique()),
                "months": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _seed_candidates(
    catalog: list[str],
    assets: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
) -> list[MonthlyMultiAssetCandidate]:
    stage_catalog = [spec for index, spec in enumerate(catalog) if index % config.total_stages == config.stage]
    if not stage_catalog:
        stage_catalog = catalog
    universes = _asset_universes(assets)
    candidates: list[MonthlyMultiAssetCandidate] = []
    exposures = (-0.8, -0.5, -0.25, 0.0, 0.25, 0.5, 0.8)
    scales = (0.25, 0.5, 0.75, 1.0)
    for spec_index, spec in enumerate(stage_catalog[: config.seed_pool]):
        universe = universes[spec_index % len(universes)]
        selector = ASSET_SELECTORS[spec_index % len(ASSET_SELECTORS)]
        for scale in scales:
            candidates.append(MonthlyMultiAssetCandidate((spec,), universe, selector=selector, scale=scale))
        candidates.append(
            MonthlyMultiAssetCandidate(
                (spec,),
                universe,
                selector=selector,
                intercept=float(rng.choice(exposures)),
                scale=0.25,
            )
        )
    while len(candidates) < config.seed_pool:
        size = int(rng.integers(1, min(config.max_features, 4) + 1))
        specs = tuple(sorted(rng.choice(stage_catalog, size=size, replace=False).tolist()))
        universe = universes[int(rng.integers(0, len(universes)))]
        candidates.append(
            MonthlyMultiAssetCandidate(
                specs,
                tuple(sorted(set(universe))),
                selector=str(rng.choice(ASSET_SELECTORS)),
                intercept=float(rng.uniform(-0.35, 0.35)),
                scale=float(rng.uniform(0.25, 1.0)),
                smoothing=float(rng.choice([0.0, 0.15, 0.30])),
            )
        )
    return candidates[: config.seed_pool]


def _evaluate_unique(
    examples: pd.DataFrame,
    candidates: list[MonthlyMultiAssetCandidate],
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str, float, float, float]],
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
        row, _, _ = evaluate_monthly_multi_asset_candidate(examples, candidate)
        rows.append(row)
    return rows


def _select_beam(rows: list[dict[str, object]], size: int) -> list[dict[str, object]]:
    by_id = {str(row["candidate_id"]): row for row in rows}
    return sorted(by_id.values(), key=lambda row: float(row["monthly_multi_asset_score"]), reverse=True)[:size]


def _mutate_candidate(
    candidate: MonthlyMultiAssetCandidate,
    catalog: list[str],
    assets: list[str],
    *,
    config: MonthlyRiskSearchConfig,
    rng: np.random.Generator,
) -> MonthlyMultiAssetCandidate:
    specs = list(candidate.specs)
    action = str(rng.choice(["add", "replace", "drop", "tweak", "selector", "assets"]))
    if action == "add" and len(specs) < config.max_features:
        specs.append(str(rng.choice(catalog)))
    elif action == "replace" and specs:
        specs[int(rng.integers(0, len(specs)))] = str(rng.choice(catalog))
    elif action == "drop" and len(specs) > 1:
        specs.pop(int(rng.integers(0, len(specs))))
    elif action == "selector":
        return MonthlyMultiAssetCandidate(
            tuple(sorted(set(specs))),
            candidate.assets,
            selector=str(rng.choice(ASSET_SELECTORS)),
            intercept=candidate.intercept,
            scale=candidate.scale,
            smoothing=candidate.smoothing,
        )
    elif action == "assets":
        return MonthlyMultiAssetCandidate(
            tuple(sorted(set(specs))),
            tuple(sorted(set(_asset_universes(assets)[int(rng.integers(0, len(_asset_universes(assets))))]))),
            selector=candidate.selector,
            intercept=candidate.intercept,
            scale=candidate.scale,
            smoothing=candidate.smoothing,
        )
    else:
        specs = _tweak_spec_weight(specs, rng)
    intercept = float(np.clip(candidate.intercept + rng.normal(0.0, 0.12), -0.8, 0.8))
    scale = float(np.clip(candidate.scale + rng.normal(0.0, 0.18), 0.05, 1.50))
    smoothing = float(np.clip(candidate.smoothing + rng.normal(0.0, 0.10), 0.0, 0.70))
    return MonthlyMultiAssetCandidate(
        tuple(sorted(set(specs))),
        candidate.assets,
        selector=candidate.selector,
        intercept=intercept,
        scale=scale,
        smoothing=smoothing,
    )


def _tweak_spec_weight(specs: list[str], rng: np.random.Generator) -> list[str]:
    if not specs:
        return specs
    index = int(rng.integers(0, len(specs)))
    parts = specs[index].split("|")
    parts[-1] = str(int(rng.choice([1, 2, 3])))
    specs[index] = "|".join(parts)
    return specs


def _candidate_from_row(row: dict[str, object] | pd.Series) -> MonthlyMultiAssetCandidate:
    specs = tuple(part for part in str(row.get("specs", "")).split(";") if part)
    assets = tuple(part for part in str(row.get("assets", "SPY")).split(",") if part)
    return MonthlyMultiAssetCandidate(
        specs,
        assets or ("SPY",),
        selector=str(row.get("asset_selector", "momentum_6m") or "momentum_6m"),
        intercept=float(row.get("intercept", 0.0) or 0.0),
        scale=float(row.get("scale", 1.0) or 1.0),
        smoothing=float(row.get("smoothing", 0.0) or 0.0),
    )


def _candidate_id(candidate: MonthlyMultiAssetCandidate) -> str:
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
    return "monthly_multi_asset_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _formula_text(candidate: MonthlyMultiAssetCandidate) -> str:
    return (
        "asset = monthly best by "
        f"{candidate.selector}; exposure = clip({candidate.intercept:.4f} + "
        f"{candidate.scale:.4f} * weighted_rule_vote, -1, 1)"
    )


def _multi_asset_score(row: dict[str, object]) -> float:
    return _monthly_risk_score(row) + float(row.get("unique_assets_used", 0) or 0) * 3.0


def _verified_train_validation_10pct(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "accepted",
        "train_years_ge_10pct",
        "train_years_total",
        "validation_years_ge_10pct",
        "validation_years_total",
        "locked_opened",
    }
    if rows.empty or not required.issubset(rows.columns):
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
    return rows.loc[mask].sort_values("monthly_multi_asset_score", ascending=False).copy()


def _summary(rows: list[dict[str, object]], *, stage: int | None = None) -> dict[str, object]:
    best = rows[0] if rows else {}
    return {
        "stage": stage,
        "candidates_evaluated": int(len(rows)),
        "accepted": int(sum(bool(row.get("accepted")) for row in rows)),
        "best_candidate": best.get("candidate_id"),
        "best_assets": best.get("assets"),
        "best_asset_selector": best.get("asset_selector"),
        "best_train_min_year_return": best.get("train_min_year_return"),
        "best_validation_min_year_return": best.get("validation_min_year_return"),
        "locked_opened": False,
        "score_mode": "train_only_monthly_multi_asset_10pct",
        "validation_role": "report_only",
    }


def _empty_summary() -> dict[str, object]:
    return {
        "rows": 0,
        "accepted": 0,
        "verified_train_validation_10pct": 0,
        "unique_verified_train_validation_10pct": 0,
        "locked_opened": False,
        "score_mode": "train_only_monthly_multi_asset_10pct",
        "validation_role": "report_only",
        "tradable_assets": list(TRADABLE_ASSETS),
        "non_tradable_indicators": list(NON_TRADABLE_INDICATORS),
    }


def _asset_col(asset: str, suffix: str) -> str:
    return f"asset_{asset.lower()}_{suffix}"


def _ratio_column(asset: str) -> str:
    return f"{asset.lower()}_close_ratio"
