from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TRAIN_END_YEAR = 2004
VALIDATION_START_YEAR = 2005
VALIDATION_END_YEAR = 2015
LOCKED_START_YEAR = 2016
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "configs" / "annual_feature_manifest.csv"


@dataclass(frozen=True)
class AnnualCandidate:
    specs: tuple[str, ...]
    min_votes: int = 1


@dataclass(frozen=True)
class AnnualBeamConfig:
    stage: int
    total_stages: int = 64
    seed_pool: int = 400
    beam_width: int = 32
    generations: int = 5
    mutations_per_parent: int = 10
    max_features: int = 4
    random_seed: int = 173_000
    score_mode: str = "validation"


def build_annual_examples(
    daily: pd.DataFrame,
    *,
    start_year: int = 1980,
    end_year: int | None = None,
) -> pd.DataFrame:
    data = _normalize_daily(daily)
    end_year = end_year or int(data.index.max().year)
    features = _build_daily_feature_frame(data)
    rows: list[dict[str, object]] = []
    for target_year in range(start_year, end_year + 1):
        previous_year = data.loc[data.index.year == target_year - 1]
        target = data.loc[data.index.year == target_year]
        if previous_year.empty or target.empty:
            continue
        decision_date = previous_year.index[-1]
        feature_row = features.loc[decision_date]
        spy_return = float(target["close"].iloc[-1] / previous_year["close"].iloc[-1] - 1.0)
        row: dict[str, object] = {
            "target_year": int(target_year),
            "decision_date": decision_date,
            "spy_return_next_year": spy_return,
            "target_positive": bool(spy_return > 0.0),
            **_political_features(target_year),
            **_calendar_cycle_features(data, target_year),
        }
        for name, value in feature_row.items():
            row[name] = float(value) if pd.notna(value) else np.nan
        rows.append(row)
    examples = pd.DataFrame(rows)
    if not examples.empty:
        examples["decision_date"] = pd.to_datetime(examples["decision_date"])
        examples["target_positive"] = examples["target_positive"].astype(bool)
    examples = _ensure_manifest_columns(examples)
    examples = _add_annual_derived_features(examples)
    return examples.replace([np.inf, -np.inf], np.nan)


def load_annual_feature_manifest(path: str | Path = MANIFEST_PATH) -> pd.DataFrame:
    manifest = pd.read_csv(path, sep=";")
    expected = {"id", "bloque", "feature", "calculo", "fuente_gratis", "serie_o_ticker", "inicio_fuente", "gratis_desde_1980", "nota"}
    missing = expected - set(manifest.columns)
    if missing:
        raise ValueError(f"annual feature manifest missing columns: {sorted(missing)}")
    return manifest


def evaluate_annual_candidate(
    examples: pd.DataFrame,
    candidate: AnnualCandidate,
    *,
    score_mode: str = "validation",
) -> dict[str, object]:
    predictions = _predict_positive(examples, candidate)
    train_mask = examples["target_year"].astype(int) <= TRAIN_END_YEAR
    validation_mask = (examples["target_year"].astype(int) >= VALIDATION_START_YEAR) & (
        examples["target_year"].astype(int) <= VALIDATION_END_YEAR
    )
    target = examples["target_positive"].astype(bool).to_numpy()
    returns = examples["spy_return_next_year"].astype(float).to_numpy()
    train = _period_metrics(predictions, target, returns, train_mask.to_numpy())
    validation = _period_metrics(predictions, target, returns, validation_mask.to_numpy())
    train_predicted_returns = _fit_predicted_returns(predictions[train_mask.to_numpy()], returns[train_mask.to_numpy()])
    train_mae = _mae_from_mapping(predictions[train_mask.to_numpy()], returns[train_mask.to_numpy()], train_predicted_returns)
    validation_mae = _mae_from_mapping(
        predictions[validation_mask.to_numpy()],
        returns[validation_mask.to_numpy()],
        train_predicted_returns,
    )
    row = {
        "candidate_id": _candidate_id(candidate),
        "specs": ";".join(candidate.specs),
        "min_votes": int(candidate.min_votes),
        "feature_count": len(candidate.specs),
        "train_hits": train["hits"],
        "train_total": train["total"],
        "train_accuracy": train["accuracy"],
        "train_negative_hits": train["negative_hits"],
        "train_negative_total": train["negative_total"],
        "validation_hits": validation["hits"],
        "validation_total": validation["total"],
        "validation_accuracy": validation["accuracy"],
        "validation_negative_hits": validation["negative_hits"],
        "validation_negative_total": validation["negative_total"],
        "train_return_mae": train_mae,
        "validation_return_mae": validation_mae,
        "always_positive_validation_accuracy": _always_positive_accuracy(examples, validation_mask),
        "annual_score": 0.0,
        "accepted": False,
        "rejection_reason": "",
        "locked_opened": False,
        "locked_hits": 0,
        "locked_total": 0,
        "score_mode": score_mode,
    }
    row["annual_score"] = score_annual_candidate(row, score_mode=score_mode)
    row["rejection_reason"] = score_annual_candidate(row, score_mode=score_mode, field="rejection_reason")
    row["accepted"] = row["rejection_reason"] == ""
    return row


def audit_annual_feature_coverage(examples: pd.DataFrame) -> pd.DataFrame:
    manifest = load_annual_feature_manifest()
    rows: list[dict[str, object]] = []
    for item in manifest.to_dict("records"):
        feature = str(item["feature"])
        series = pd.to_numeric(examples.get(feature, pd.Series(index=examples.index, dtype=float)), errors="coerce")
        non_null = series.notna()
        years = examples.loc[non_null, "target_year"].astype(int) if "target_year" in examples else pd.Series(dtype=int)
        train = examples["target_year"].astype(int) <= TRAIN_END_YEAR
        validation = (examples["target_year"].astype(int) >= VALIDATION_START_YEAR) & (
            examples["target_year"].astype(int) <= VALIDATION_END_YEAR
        )
        usable = _feature_usable_for_beam(series, examples)
        rows.append(
            {
                **item,
                "non_null_years": int(non_null.sum()),
                "total_years": int(len(examples)),
                "coverage": float(non_null.mean()) if len(examples) else 0.0,
                "first_usable_year": int(years.min()) if not years.empty else None,
                "last_usable_year": int(years.max()) if not years.empty else None,
                "train_non_null_years": int(non_null[train].sum()),
                "validation_non_null_years": int(non_null[validation].sum()),
                "unique_train_values": int(series[train].nunique(dropna=True)),
                "quality": _feature_quality(item, non_null, examples),
                "usable_in_beam": bool(usable),
            }
        )
    return pd.DataFrame(rows)


def annual_score(row: dict[str, object]) -> float:
    return score_annual_candidate(row, score_mode="validation")


def score_annual_candidate(
    row: dict[str, object],
    *,
    score_mode: str = "validation",
    field: str = "score",
) -> float | str:
    if score_mode == "train_only_100":
        if field == "rejection_reason":
            return _train_only_100_rejection_reason(row)
        return _train_only_100_score(row)
    if score_mode == "train_validation_100":
        if field == "rejection_reason":
            return _train_validation_100_rejection_reason(row)
        return _train_only_100_score(row)
    if score_mode != "validation":
        raise ValueError(f"unknown annual score mode: {score_mode}")
    if field == "rejection_reason":
        return _rejection_reason(row)
    return _validation_score(row)


def _validation_score(row: dict[str, object]) -> float:
    validation_accuracy = float(row.get("validation_accuracy", 0.0) or 0.0)
    train_accuracy = float(row.get("train_accuracy", 0.0) or 0.0)
    validation_negative_hits = int(row.get("validation_negative_hits", 0) or 0)
    train_negative_hits = int(row.get("train_negative_hits", 0) or 0)
    validation_mae = float(row.get("validation_return_mae", 1.0) or 1.0)
    complexity_penalty = max(0, int(row.get("feature_count", 1) or 1) - 2) * 0.5
    baseline = float(row.get("always_positive_validation_accuracy", 0.0) or 0.0)
    baseline_bonus = max(0.0, validation_accuracy - baseline) * 20.0
    return float(
        validation_accuracy * 100.0
        + train_accuracy * 30.0
        + validation_negative_hits * 8.0
        + train_negative_hits * 2.0
        + baseline_bonus
        - validation_mae * 10.0
        - complexity_penalty
    )


def _train_only_100_score(row: dict[str, object]) -> float:
    train_accuracy = float(row.get("train_accuracy", 0.0) or 0.0)
    train_hits = int(row.get("train_hits", 0) or 0)
    train_negative_hits = int(row.get("train_negative_hits", 0) or 0)
    train_negative_total = int(row.get("train_negative_total", 0) or 0)
    train_mae = float(row.get("train_return_mae", 1.0) or 1.0)
    feature_count = int(row.get("feature_count", 1) or 1)
    perfect_bonus = 5_000.0 if train_accuracy >= 1.0 else 0.0
    stress_bonus = 250.0 if train_negative_total and train_negative_hits == train_negative_total else 0.0
    return float(
        perfect_bonus
        + train_accuracy * 1_000.0
        + train_hits * 10.0
        + train_negative_hits * 35.0
        + stress_bonus
        - feature_count * 8.0
        - train_mae * 10.0
    )


def run_annual_beam_search(
    examples: pd.DataFrame,
    config: AnnualBeamConfig,
) -> list[dict[str, object]]:
    catalog = _build_spec_catalog(examples)
    if not catalog:
        return []
    rng = np.random.default_rng(config.random_seed + config.stage)
    rows: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, ...], int]] = set()
    seeds = _seed_candidates(catalog, config=config, rng=rng)
    seed_rows = _evaluate_unique(examples, seeds, seen=seen, score_mode=config.score_mode)
    rows.extend(seed_rows)
    beam = _select_beam(seed_rows, config.beam_width)
    for _ in range(config.generations):
        children: list[AnnualCandidate] = []
        for parent in beam:
            candidate = _candidate_from_row(parent)
            for _ in range(config.mutations_per_parent):
                children.append(_mutate_candidate(candidate, catalog, config=config, rng=rng))
        child_rows = _evaluate_unique(examples, children, seen=seen, score_mode=config.score_mode)
        rows.extend(child_rows)
        beam = _select_beam([*beam, *child_rows], config.beam_width)
    return sorted(rows, key=lambda row: float(row["annual_score"]), reverse=True)


def _normalize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily.copy()
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.set_index("timestamp")
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    if "close" not in data.columns:
        raise ValueError("annual prediction data needs a close column")
    return data


def _build_daily_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(data["close"], errors="coerce")
    daily_return = close.pct_change()
    sma_50 = close.rolling(50, min_periods=20).mean()
    sma_100 = close.rolling(100, min_periods=50).mean()
    sma_200 = close.rolling(200, min_periods=100).mean()
    realized_vol_21d = daily_return.rolling(21, min_periods=10).std(ddof=0) * np.sqrt(252.0)
    out: dict[str, pd.Series] = {
        "sp500_return_21d": close.pct_change(21),
        "sp500_return_63d": close.pct_change(63),
        "sp500_return_126d": close.pct_change(126),
        "sp500_return_252d": close.pct_change(252),
        "sp500_return_504d": close.pct_change(504),
        "sp500_above_sma_50": (close > sma_50).astype(float),
        "sp500_above_sma_100": (close > sma_100).astype(float),
        "sp500_above_sma_200": (close > sma_200).astype(float),
        "sp500_distance_sma_50": close / sma_50 - 1.0,
        "sp500_distance_sma_200": close / sma_200 - 1.0,
        "sp500_52w_high_distance": close / close.rolling(252, min_periods=63).max() - 1.0,
        "sp500_drawdown_current": close / close.cummax() - 1.0,
        "realized_vol_21d": realized_vol_21d,
        "realized_vol_63d": daily_return.rolling(63, min_periods=21).std(ddof=0) * np.sqrt(252.0),
        "realized_vol_126d": daily_return.rolling(126, min_periods=42).std(ddof=0) * np.sqrt(252.0),
        "realized_vol_252d": daily_return.rolling(252, min_periods=63).std(ddof=0) * np.sqrt(252.0),
        "volatility_spike_dummy": (
            realized_vol_21d > realized_vol_21d.rolling(252, min_periods=63).quantile(0.95)
        ).astype(float),
        "spy_return_3m": close.pct_change(63),
        "spy_return_6m": close.pct_change(126),
        "spy_return_12m": close.pct_change(252),
        "spy_return_36m": close.pct_change(756),
        "spy_vol_3m": close.pct_change().rolling(63, min_periods=21).std(ddof=0),
        "spy_vol_12m": close.pct_change().rolling(252, min_periods=63).std(ddof=0),
        "spy_drawdown_12m": close / close.rolling(252, min_periods=63).max() - 1.0,
    }
    _add_named_feature_aliases(out, data, close)
    skip = {"open", "high", "low", "close", "volume"}
    for column in data.columns:
        if column in skip:
            continue
        series = pd.to_numeric(data[column], errors="coerce")
        if series.notna().sum() < 120:
            continue
        out[column] = series
        out[f"{column}_change_12m"] = series.diff(252)
        mean = series.rolling(756, min_periods=126).mean()
        std = series.rolling(756, min_periods=126).std(ddof=0).replace(0, np.nan)
        out[f"{column}_z_3y"] = (series - mean) / std
    return pd.DataFrame(out, index=data.index)


def _add_named_feature_aliases(out: dict[str, pd.Series], data: pd.DataFrame, close: pd.Series) -> None:
    aliases = {
        "vix_level": "vix_level",
        "vix3m_level": "vix3m_level",
        "cape_ratio": "cape",
        "pe_ttm": "pe_ttm",
        "earnings_yield": "earnings_yield",
        "dividend_yield": "dividend_yield",
        "buyback_yield": "buyback_yield",
        "market_cap_to_gdp": "market_cap_to_gdp",
        "fed_funds_rate": "fed_funds",
        "treasury_3m": "treasury_3m",
        "treasury_2y": "yield_2y",
        "yield_5y": "yield_5y",
        "treasury_10y": "yield_10y",
        "treasury_30y": "yield_30y",
        "high_yield_oas": "hy_oas",
        "investment_grade_oas": "ig_oas",
        "baa_aaa_spread": "baa_aaa_spread",
        "excess_bond_premium": "excess_bond_premium",
        "consumer_confidence": "consumer_confidence",
        "unemployment_rate": "unemployment",
        "ism_manufacturing": "ism_manufacturing",
        "recession_dummy": "recession_dummy",
        "money_market_assets": "money_market_assets",
        "financial_conditions_index": "financial_conditions_index",
        "stl_fed_stress_index": "financial_stress",
    }
    for target, source in aliases.items():
        if source in data:
            out[target] = pd.to_numeric(data[source], errors="coerce")
    if "vix_level" in out:
        out["vix_change_21d"] = out["vix_level"].diff(21)
        out["vix_percentile_252d"] = _rolling_percentile(out["vix_level"], 252)
        out["realized_vol_minus_vix"] = out["realized_vol_21d"] - out["vix_level"] / 100.0
    if "vix3m_level" in out and "vix_level" in out:
        out["vix_term_structure"] = out["vix3m_level"] / out["vix_level"]
    if "cape_ratio" in out:
        out["cape_percentile"] = _expanding_percentile(out["cape_ratio"])
        out["cape_minus_10y_yield"] = 1.0 / out["cape_ratio"] - _rate_to_decimal(out.get("treasury_10y"))
    if "earnings_yield" in out and "treasury_10y" in out:
        out["equity_risk_premium"] = out["earnings_yield"] - _rate_to_decimal(out["treasury_10y"])
        out["earnings_yield_minus_10y_yield"] = out["earnings_yield"] - _rate_to_decimal(out["treasury_10y"])
    if "dividend_yield" in out and "buyback_yield" in out:
        out["shareholder_yield"] = out["dividend_yield"] + out["buyback_yield"]
    _add_macro_features(out, data)
    _add_rate_features(out)
    _add_credit_features(out)
    _add_external_features(out, data, close)


def _add_rate_features(out: dict[str, pd.Series]) -> None:
    fed = out.get("fed_funds_rate")
    ten = out.get("treasury_10y")
    three_month = out.get("treasury_3m")
    two = out.get("treasury_2y")
    thirty = out.get("treasury_30y")
    five = out.get("yield_5y")
    cpi_yoy = out.get("cpi_yoy")
    if fed is not None:
        out["fed_funds_change_252d"] = fed.diff(252)
        out["rate_hike_dummy"] = (fed.diff(252) > 0).astype(float)
        out["rate_cut_dummy"] = (fed.diff(252) < 0).astype(float)
        if cpi_yoy is not None:
            out["real_fed_funds"] = fed - cpi_yoy * 100.0
    if ten is not None:
        if cpi_yoy is not None:
            out["real_10y_yield"] = ten - cpi_yoy * 100.0
        if three_month is not None:
            out["spread_10y_3m"] = ten - three_month
            out["yield_curve_inverted_dummy"] = (out["spread_10y_3m"] < 0).astype(float)
            out["months_since_inversion"] = _periods_since(out["spread_10y_3m"] < 0)
            out["nyfed_recession_probability"] = _nyfed_recession_probability(out["spread_10y_3m"])
        if two is not None:
            out["spread_10y_2y"] = ten - two
            out["curve_steepening_63d"] = out["spread_10y_2y"].diff(63)
            out["curve_steepening_252d"] = out["spread_10y_2y"].diff(252)
    if thirty is not None and five is not None:
        out["spread_30y_5y"] = thirty - five


def _add_credit_features(out: dict[str, pd.Series]) -> None:
    spread = out.get("baa_aaa_spread")
    if spread is None:
        return
    out["credit_spread_change_21d"] = spread.diff(21)
    out["credit_spread_change_63d"] = spread.diff(63)
    out["credit_spread_percentile_252d"] = _rolling_percentile(spread, 252)
    out["credit_stress_dummy"] = (spread > spread.rolling(252, min_periods=63).quantile(0.90)).astype(float)


def _add_macro_features(out: dict[str, pd.Series], data: pd.DataFrame) -> None:
    cpi = pd.to_numeric(data["cpi"], errors="coerce") if "cpi" in data else None
    core_cpi = pd.to_numeric(data["core_cpi"], errors="coerce") if "core_cpi" in data else None
    pce = pd.to_numeric(data["pce"], errors="coerce") if "pce" in data else None
    core_pce = pd.to_numeric(data["core_pce"], errors="coerce") if "core_pce" in data else None
    if cpi is not None:
        out["cpi_yoy"] = cpi.pct_change(252)
        out["inflation_change_6m"] = out["cpi_yoy"] - out["cpi_yoy"].shift(126)
    if core_cpi is not None:
        out["core_cpi_yoy"] = core_cpi.pct_change(252)
    if pce is not None:
        out["pce_yoy"] = pce.pct_change(252)
    if core_pce is not None:
        out["core_pce_yoy"] = core_pce.pct_change(252)
    if "industrial_production" in data:
        out["industrial_production_yoy"] = pd.to_numeric(data["industrial_production"], errors="coerce").pct_change(252)
    if "real_gdp" in data:
        out["real_gdp_yoy"] = pd.to_numeric(data["real_gdp"], errors="coerce").pct_change(252)
    if "unemployment" in data:
        unrate = pd.to_numeric(data["unemployment"], errors="coerce")
        out["unemployment_rate"] = unrate
        out["unemployment_change_6m"] = unrate - unrate.shift(126)
    if "initial_claims" in data:
        out["initial_claims_4w_avg"] = pd.to_numeric(data["initial_claims"], errors="coerce").rolling(20, min_periods=4).mean()
    if "payrolls" in data:
        payrolls = pd.to_numeric(data["payrolls"], errors="coerce")
        out["payrolls_3m_avg"] = payrolls.diff(21).rolling(63, min_periods=21).mean()
    if "retail_sales_yoy_source" in data:
        out["retail_sales_yoy"] = pd.to_numeric(data["retail_sales_yoy_source"], errors="coerce").pct_change(252)
    if "gasoline_yoy_source" in data:
        out["gasoline_yoy"] = pd.to_numeric(data["gasoline_yoy_source"], errors="coerce").pct_change(252)
    if "lei" in data:
        out["lei_yoy"] = pd.to_numeric(data["lei"], errors="coerce").pct_change(252)
    if "eps_ttm" in data:
        out["eps_ttm_growth"] = pd.to_numeric(data["eps_ttm"], errors="coerce").pct_change(252)
    if "m2" in data:
        out["m2_yoy"] = pd.to_numeric(data["m2"], errors="coerce").pct_change(252)
    if "fed_balance_sheet" in data:
        out["fed_balance_sheet_yoy"] = pd.to_numeric(data["fed_balance_sheet"], errors="coerce").pct_change(252)
    if "commercial_bank_credit" in data:
        out["commercial_bank_credit_yoy"] = pd.to_numeric(data["commercial_bank_credit"], errors="coerce").pct_change(252)
    if {"m2", "fed_balance_sheet", "treasury_general_account", "reverse_repo_level"}.issubset(data.columns):
        out["dollar_liquidity_proxy"] = (
            pd.to_numeric(data["m2"], errors="coerce")
            + pd.to_numeric(data["fed_balance_sheet"], errors="coerce")
            - pd.to_numeric(data["treasury_general_account"], errors="coerce")
            - pd.to_numeric(data["reverse_repo_level"], errors="coerce")
        )


def _add_external_features(out: dict[str, pd.Series], data: pd.DataFrame, close: pd.Series) -> None:
    if "dxy_proxy" in data:
        dxy = pd.to_numeric(data["dxy_proxy"], errors="coerce")
        out["dxy_return_252d"] = dxy.pct_change(252)
        out["dxy_change_63d"] = dxy.diff(63)
    if "gold" in data:
        out["gold_return_252d"] = pd.to_numeric(data["gold"], errors="coerce").pct_change(252)
    if "oil" in data:
        out["oil_yoy"] = pd.to_numeric(data["oil"], errors="coerce").pct_change(252)
        out["oil_return_252d"] = out["oil_yoy"]
    if "copper" in data:
        out["copper_return_252d"] = pd.to_numeric(data["copper"], errors="coerce").pct_change(252)
        if "gold" in data:
            out["copper_gold_ratio"] = pd.to_numeric(data["copper"], errors="coerce") / pd.to_numeric(data["gold"], errors="coerce")
    if "treasury_10y" in out:
        duration = 7.0
        rate = _rate_to_decimal(out["treasury_10y"])
        out["us_bonds_return_252d"] = rate.shift(252) - duration * rate.diff(252)
        out["stocks_vs_bonds_252d"] = close.pct_change(252) - out["us_bonds_return_252d"]


def _ensure_manifest_columns(examples: pd.DataFrame) -> pd.DataFrame:
    missing = {}
    for feature in load_annual_feature_manifest()["feature"]:
        if feature not in examples.columns:
            missing[feature] = np.nan
    if missing:
        examples = pd.concat([examples, pd.DataFrame(missing, index=examples.index)], axis=1)
    return examples.copy()


def _add_annual_derived_features(examples: pd.DataFrame) -> pd.DataFrame:
    if examples.empty:
        return examples
    reserved = {"target_year", "decision_date", "spy_return_next_year", "target_positive"}
    output = examples.copy()
    derived: dict[str, pd.Series] = {}
    ordered = output.sort_values("target_year").reset_index(drop=True)
    for column in list(output.columns):
        if column in reserved or column.endswith(("_annual_change_1y", "_annual_change_3y", "_annual_percentile")):
            continue
        series = pd.to_numeric(ordered[column], errors="coerce")
        if series.notna().sum() < 4 or series.nunique(dropna=True) < 3:
            continue
        derived[f"{column}_annual_change_1y"] = series.diff(1)
        derived[f"{column}_annual_change_3y"] = series.diff(3)
        derived[f"{column}_annual_percentile"] = _expanding_percentile(series)
    if not derived:
        return output
    derived_frame = pd.DataFrame(derived)
    derived_frame.index = ordered.index
    enriched = pd.concat([ordered, derived_frame], axis=1)
    return enriched.sort_values("target_year").reset_index(drop=True)


def _political_features(target_year: int) -> dict[str, float]:
    cycle_year = ((target_year - 1981) % 4) + 1
    president_party = _president_party(target_year)
    house_party = _house_party(target_year)
    senate_party = _senate_party(target_year)
    return {
        "presidential_cycle_year": float(cycle_year),
        "is_election_year": float(cycle_year == 4),
        "is_post_election_year": float(cycle_year == 1),
        "midterm_year_dummy": float(cycle_year == 2),
        "election_year_dummy": float(cycle_year == 4),
        "president_party": float(president_party),
        "house_control_party": float(house_party),
        "senate_control_party": float(senate_party),
        "split_congress": float(house_party != senate_party),
        "unified_government": float(house_party == senate_party == president_party),
    }


def _calendar_cycle_features(data: pd.DataFrame, target_year: int) -> dict[str, float]:
    source_year = target_year - 1
    source = data.loc[data.index.year == source_year]
    prior = data.loc[data.index.year == source_year - 1]
    features = {
        "month_number": 12.0,
        "quarter_number": 4.0,
        "january_return": np.nan,
        "first_5_days_return": np.nan,
        "santa_rally_return": np.nan,
        "year_after_negative_year": np.nan,
        "consecutive_positive_years": np.nan,
    }
    if not source.empty:
        january = source.loc[source.index.month == 1]
        if len(january) >= 2:
            features["january_return"] = float(january["close"].iloc[-1] / january["close"].iloc[0] - 1.0)
        first_days = source.iloc[:5]
        if len(first_days) >= 2:
            features["first_5_days_return"] = float(first_days["close"].iloc[-1] / first_days["close"].iloc[0] - 1.0)
    if not prior.empty and not source.empty:
        santa_piece = pd.concat([prior.iloc[-5:], source.iloc[:2]])
        if len(santa_piece) >= 2:
            features["santa_rally_return"] = float(santa_piece["close"].iloc[-1] / santa_piece["close"].iloc[0] - 1.0)
    features["year_after_negative_year"] = float(_calendar_year_return(data, source_year) < 0.0)
    count = 0
    for year in range(source_year, int(data.index.min().year) - 1, -1):
        year_return = _calendar_year_return(data, year)
        if pd.isna(year_return) or year_return <= 0.0:
            break
        count += 1
    features["consecutive_positive_years"] = float(count)
    return features


def _calendar_year_return(data: pd.DataFrame, year: int) -> float:
    frame = data.loc[data.index.year == year]
    if len(frame) < 2:
        return np.nan
    return float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1.0)


def _president_party(year: int) -> int:
    # Republican = 1, Democrat = -1. Party for the president serving most of the target year.
    republican_ranges = ((1981, 1992), (2001, 2008), (2017, 2020), (2025, 2028))
    return 1 if any(start <= year <= end for start, end in republican_ranges) else -1


def _house_party(year: int) -> int:
    republican_ranges = ((1995, 2006), (2011, 2018), (2023, 2026))
    return 1 if any(start <= year <= end for start, end in republican_ranges) else -1


def _senate_party(year: int) -> int:
    republican_ranges = ((1981, 1986), (1995, 2001), (2003, 2006), (2015, 2020), (2025, 2026))
    return 1 if any(start <= year <= end for start, end in republican_ranges) else -1


def _build_spec_catalog(examples: pd.DataFrame) -> list[str]:
    train = examples.loc[examples["target_year"].astype(int) <= TRAIN_END_YEAR]
    reserved = {"target_year", "decision_date", "spy_return_next_year", "target_positive"}
    specs: list[str] = []
    for column in examples.columns:
        if column in reserved:
            continue
        series = pd.to_numeric(train[column], errors="coerce").dropna()
        if not _feature_usable_for_beam(examples[column], examples):
            continue
        for quantile in (0.25, 0.50, 0.75):
            value = float(series.quantile(quantile))
            if not isfinite(value):
                continue
            specs.append(_encode_spec(column, value, 1))
            specs.append(_encode_spec(column, value, -1))
            if quantile == 0.50:
                specs.append(_encode_weighted_threshold_spec(column, value, 1, 2))
                specs.append(_encode_weighted_threshold_spec(column, value, -1, 2))
        for lower_quantile, upper_quantile in ((0.20, 0.80), (0.25, 0.75), (0.33, 0.67)):
            lower = float(series.quantile(lower_quantile))
            upper = float(series.quantile(upper_quantile))
            if not isfinite(lower) or not isfinite(upper) or lower >= upper:
                continue
            specs.append(_encode_range_spec(column, lower, upper, 1))
            specs.append(_encode_range_spec(column, lower, upper, -1))
            if lower_quantile == 0.25:
                specs.append(_encode_weighted_range_spec(column, lower, upper, 1, 2))
    return sorted(set(specs))


def _feature_usable_for_beam(series: pd.Series, examples: pd.DataFrame) -> bool:
    train = examples["target_year"].astype(int) <= TRAIN_END_YEAR
    values = pd.to_numeric(series, errors="coerce")
    train_values = values[train].dropna()
    return len(train_values) >= 8 and train_values.nunique() >= 4


def _feature_quality(item: dict[str, object], non_null: pd.Series, examples: pd.DataFrame) -> str:
    note = str(item.get("nota", "")).lower()
    free_since_1980 = str(item.get("gratis_desde_1980", "")).lower()
    source = str(item.get("serie_o_ticker", "")).lower()
    if int(non_null.sum()) == 0:
        if "no free public series" in source or "pago" in note or "no gratis fiable" in note:
            return "sin_datos_publicos_fiables"
        return "sin_datos"
    first_year = int(examples.loc[non_null, "target_year"].astype(int).min())
    if "proxy" in free_since_1980 or "proxy" in note:
        return "proxy"
    if "no" in free_since_1980 or first_year > 1980:
        return "parcial"
    return "buena"


def _seed_candidates(
    catalog: list[str],
    *,
    config: AnnualBeamConfig,
    rng: np.random.Generator,
) -> list[AnnualCandidate]:
    candidates: list[AnnualCandidate] = []
    offset_catalog = [spec for index, spec in enumerate(catalog) if index % config.total_stages == config.stage]
    base = offset_catalog or catalog
    for spec in base[: config.seed_pool]:
        candidates.append(AnnualCandidate((spec,), min_votes=1))
    while len(candidates) < config.seed_pool:
        size = int(rng.integers(1, config.max_features + 1))
        specs = tuple(sorted(rng.choice(catalog, size=size, replace=False).tolist()))
        min_votes = int(rng.integers(1, _candidate_weight_total(specs) + 1))
        candidates.append(AnnualCandidate(specs, min_votes=min_votes))
    return candidates


def _evaluate_unique(
    examples: pd.DataFrame,
    candidates: Iterable[AnnualCandidate],
    *,
    seen: set[tuple[tuple[str, ...], int]],
    score_mode: str = "validation",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        specs = tuple(sorted(set(candidate.specs)))[: max(1, len(candidate.specs))]
        clean = AnnualCandidate(specs, max(1, min(candidate.min_votes, _candidate_weight_total(specs))))
        key = (clean.specs, clean.min_votes)
        if key in seen or not clean.specs:
            continue
        seen.add(key)
        rows.append(evaluate_annual_candidate(examples, clean, score_mode=score_mode))
    return rows


def _select_beam(rows: list[dict[str, object]], width: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    signatures: set[str] = set()
    for row in sorted(rows, key=lambda item: float(item["annual_score"]), reverse=True):
        signature = _signature(str(row["specs"]))
        if signature in signatures:
            continue
        signatures.add(signature)
        selected.append(row)
        if len(selected) >= width:
            break
    return selected


def _mutate_candidate(
    candidate: AnnualCandidate,
    catalog: list[str],
    *,
    config: AnnualBeamConfig,
    rng: np.random.Generator,
) -> AnnualCandidate:
    specs = list(candidate.specs)
    action = str(rng.choice(["replace", "replace", "add", "remove", "votes", "weight"]))
    if action == "replace" and specs:
        specs[int(rng.integers(0, len(specs)))] = str(catalog[int(rng.integers(0, len(catalog)))])
    elif action == "add" and len(specs) < config.max_features:
        specs.append(str(catalog[int(rng.integers(0, len(catalog)))]))
    elif action == "remove" and len(specs) > 1:
        del specs[int(rng.integers(0, len(specs)))]
    elif action == "weight" and specs:
        index = int(rng.integers(0, len(specs)))
        specs[index] = _set_spec_weight(specs[index], int(rng.choice([1, 2, 3])))
    min_votes = candidate.min_votes
    if action == "votes":
        min_votes = int(rng.integers(1, _candidate_weight_total(specs) + 1))
    return AnnualCandidate(tuple(sorted(set(specs)))[: config.max_features], min_votes=min_votes)


def _candidate_from_row(row: dict[str, object]) -> AnnualCandidate:
    specs = tuple(part for part in str(row["specs"]).split(";") if part)
    return AnnualCandidate(specs, int(row.get("min_votes", 1) or 1))


def _predict_positive(examples: pd.DataFrame, candidate: AnnualCandidate) -> np.ndarray:
    votes = np.zeros(len(examples), dtype=float)
    for spec in candidate.specs:
        vote, weight = _spec_vote_and_weight(examples, spec)
        votes += np.nan_to_num(vote, nan=False).astype(float) * weight
    return votes >= max(1, min(candidate.min_votes, _candidate_weight_total(candidate.specs)))


def _period_metrics(
    predictions: np.ndarray,
    target: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    if not mask.any():
        return {"hits": 0, "total": 0, "accuracy": 0.0, "negative_hits": 0, "negative_total": 0}
    period_predictions = predictions[mask]
    period_target = target[mask]
    hits = period_predictions == period_target
    negative = returns[mask] < 0
    return {
        "hits": int(hits.sum()),
        "total": int(mask.sum()),
        "accuracy": float(hits.mean()),
        "negative_hits": int((hits & negative).sum()),
        "negative_total": int(negative.sum()),
    }


def _fit_predicted_returns(predictions: np.ndarray, returns: np.ndarray) -> dict[bool, float]:
    default = float(np.nanmean(returns)) if len(returns) else 0.0
    mapping = {True: default, False: default}
    for value in (True, False):
        subset = returns[predictions == value]
        if len(subset):
            mapping[value] = float(np.nanmean(subset))
    return mapping


def _mae_from_mapping(
    predictions: np.ndarray,
    returns: np.ndarray,
    mapping: dict[bool, float],
) -> float:
    if len(returns) == 0:
        return 1.0
    predicted = np.array([mapping[bool(value)] for value in predictions], dtype=float)
    return float(np.nanmean(np.abs(predicted - returns)))


def _always_positive_accuracy(examples: pd.DataFrame, mask: pd.Series) -> float:
    if not mask.any():
        return 0.0
    return float(examples.loc[mask, "target_positive"].astype(bool).mean())


def _rejection_reason(row: dict[str, object]) -> str:
    if int(row["validation_total"]) < 8:
        return "too_few_validation_years"
    if float(row["validation_accuracy"]) < 0.70:
        return "validation_accuracy"
    if float(row["train_accuracy"]) < 0.70:
        return "train_accuracy"
    if float(row["validation_accuracy"]) <= float(row["always_positive_validation_accuracy"]):
        return "baseline"
    if int(row["validation_negative_total"]) and int(row["validation_negative_hits"]) < 1:
        return "misses_validation_stress"
    if int(row["feature_count"]) > 4:
        return "too_many_features"
    return ""


def _train_only_100_rejection_reason(row: dict[str, object]) -> str:
    if int(row.get("train_total", 0) or 0) < 20:
        return "too_few_train_years"
    if float(row.get("train_accuracy", 0.0) or 0.0) < 1.0:
        return "train_not_perfect"
    if int(row.get("train_negative_total", 0) or 0) and int(row.get("train_negative_hits", 0) or 0) < int(
        row.get("train_negative_total", 0) or 0
    ):
        return "misses_train_stress"
    if int(row.get("feature_count", 1) or 1) > 5:
        return "too_many_features"
    return ""


def _train_validation_100_rejection_reason(row: dict[str, object]) -> str:
    train_reason = _train_only_100_rejection_reason(row)
    if train_reason:
        return train_reason
    if int(row.get("validation_total", 0) or 0) < 8:
        return "too_few_validation_years"
    if float(row.get("validation_accuracy", 0.0) or 0.0) < 1.0:
        return "validation_not_perfect"
    if int(row.get("validation_negative_total", 0) or 0) and int(row.get("validation_negative_hits", 0) or 0) < int(
        row.get("validation_negative_total", 0) or 0
    ):
        return "misses_validation_stress"
    return ""


def _encode_spec(feature: str, threshold: float, direction: int) -> str:
    return f"{feature}|threshold|{threshold:.8g}|{int(direction)}"


def _encode_weighted_threshold_spec(feature: str, threshold: float, direction: int, weight: int) -> str:
    return f"{feature}|threshold|{threshold:.8g}|{int(direction)}|{int(weight)}"


def _encode_range_spec(feature: str, lower: float, upper: float, direction: int) -> str:
    return f"{feature}|range|{lower:.8g}|{upper:.8g}|{int(direction)}"


def _encode_weighted_range_spec(feature: str, lower: float, upper: float, direction: int, weight: int) -> str:
    return f"{feature}|range|{lower:.8g}|{upper:.8g}|{int(direction)}|{int(weight)}"


def _decode_threshold_spec(spec: str) -> tuple[str, float, int, int]:
    parts = spec.split("|")
    if len(parts) not in {4, 5} or parts[1] != "threshold":
        raise ValueError(f"invalid annual spec: {spec}")
    weight = int(float(parts[4])) if len(parts) == 5 else 1
    return parts[0], float(parts[2]), int(float(parts[3])), max(1, weight)


def _decode_range_spec(spec: str) -> tuple[str, float, float, int, int]:
    parts = spec.split("|")
    if len(parts) not in {5, 6} or parts[1] != "range":
        raise ValueError(f"invalid annual spec: {spec}")
    weight = int(float(parts[5])) if len(parts) == 6 else 1
    return parts[0], float(parts[2]), float(parts[3]), int(float(parts[4])), max(1, weight)


def _spec_vote_and_weight(examples: pd.DataFrame, spec: str) -> tuple[np.ndarray, int]:
    parts = spec.split("|")
    if len(parts) < 2:
        raise ValueError(f"invalid annual spec: {spec}")
    if parts[1] == "threshold":
        feature, threshold, direction, weight = _decode_threshold_spec(spec)
        values = pd.to_numeric(examples[feature], errors="coerce").to_numpy(dtype=float)
        vote = values > threshold if direction >= 0 else values < threshold
        return vote, weight
    if parts[1] == "range":
        feature, lower, upper, direction, weight = _decode_range_spec(spec)
        values = pd.to_numeric(examples[feature], errors="coerce").to_numpy(dtype=float)
        inside = (values >= lower) & (values <= upper)
        vote = inside if direction >= 0 else ~inside
        return vote, weight
    raise ValueError(f"invalid annual spec: {spec}")


def _spec_weight(spec: str) -> int:
    parts = spec.split("|")
    if len(parts) >= 5 and parts[1] == "threshold":
        return max(1, int(float(parts[4])))
    if len(parts) >= 6 and parts[1] == "range":
        return max(1, int(float(parts[5])))
    return 1


def _candidate_weight_total(specs: Iterable[str]) -> int:
    return max(1, sum(_spec_weight(spec) for spec in specs))


def _set_spec_weight(spec: str, weight: int) -> str:
    parts = spec.split("|")
    if len(parts) == 4 and parts[1] == "threshold":
        return "|".join([*parts, str(weight)])
    if len(parts) == 5 and parts[1] == "threshold":
        return "|".join([*parts[:4], str(weight)])
    if len(parts) == 5 and parts[1] == "range":
        return "|".join([*parts, str(weight)])
    if len(parts) == 6 and parts[1] == "range":
        return "|".join([*parts[:5], str(weight)])
    raise ValueError(f"invalid annual spec: {spec}")


def _candidate_id(candidate: AnnualCandidate) -> str:
    raw = f"annual_beam|min_votes={candidate.min_votes}|specs={';'.join(candidate.specs)}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"annual_beam_{digest}"


def _signature(specs: str) -> str:
    names = sorted({part.split("|", 1)[0] for part in specs.split(";") if part})
    return ";".join(names[:4])


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(20, window // 4)).rank(pct=True)


def _expanding_percentile(series: pd.Series) -> pd.Series:
    return series.expanding(min_periods=20).rank(pct=True)


def _rate_to_decimal(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce") / 100.0


def _periods_since(condition: pd.Series) -> pd.Series:
    output = []
    last_seen = None
    for index, value in enumerate(condition.fillna(False).astype(bool)):
        if value:
            last_seen = index
            output.append(0.0)
        elif last_seen is None:
            output.append(np.nan)
        else:
            output.append(float(index - last_seen))
    return pd.Series(output, index=condition.index)


def _nyfed_recession_probability(spread_10y_3m: pd.Series) -> pd.Series:
    # Simple public-parameter proxy: inverted/flat curves imply higher 12m recession risk.
    spread = pd.to_numeric(spread_10y_3m, errors="coerce")
    z = -0.6 - 0.9 * spread
    return 1.0 / (1.0 + np.exp(-z))
