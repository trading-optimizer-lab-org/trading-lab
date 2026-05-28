from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from aurora.validation.deflated_sharpe import deflated_sharpe_annualized
from aurora.validation.robustness_config import UniversalRobustnessConfig
from aurora.validation.robustness_data_quality import validate_return_frame
from aurora.validation.robustness_duplicates import assign_duplicate_groups


@dataclass(frozen=True)
class UniversalRobustnessResult:
    result: dict[str, Any]
    bootstrap: pd.DataFrame
    year_by_year: pd.DataFrame
    data_quality: dict[str, Any]


def run_universal_robustness(
    returns: pd.DataFrame,
    config: UniversalRobustnessConfig | None = None,
) -> UniversalRobustnessResult:
    """Run the universal robustness test for one candidate return series."""

    config = config or UniversalRobustnessConfig()
    frame = _normalise_returns(returns)
    if frame.empty:
        return UniversalRobustnessResult(
            result=_empty_result(),
            bootstrap=pd.DataFrame(),
            year_by_year=pd.DataFrame(),
            data_quality={"data_quality_pass": False, "fail_reasons": "empty_returns"},
        )

    candidate_id = str(frame["candidate_id"].iloc[0])
    run_id = str(frame["run_id"].iloc[0])
    source_method = str(frame["source_method"].iloc[0])
    frequency = str(frame["frequency"].iloc[0]).lower()
    sample_role = _sample_role_summary(frame["sample_role"])
    ppy = config.periods_per_year(frequency)
    values = pd.to_numeric(frame["strategy_return"], errors="coerce").to_numpy(dtype=float)

    data_quality = validate_return_frame(frame, config)
    metrics = _return_metrics(values, ppy)
    year_by_year = _year_by_year(frame)
    bootstrap_df, bootstrap_summary = _bootstrap_metrics(values, frequency, config)
    split = _split_checks(values, ppy, config)
    looy = _leave_one_year_out(frame, ppy, config)
    calendar = _calendar_profit_checks(frame, config)
    multiple = _multiple_testing_check(frame, source_method, config)
    psr_dsr = _psr_dsr_check(values, ppy, frame, config, multiple.get("n_trials"))
    cost = _cost_check(frame, ppy, config)
    benchmark = _benchmark_checks(frame, config)

    bootstrap_pass = (
        bootstrap_summary["bootstrap_prob_cagr_positive"] >= config.prob_cagr_positive_min
        and bootstrap_summary["bootstrap_cagr_p05"] >= config.bootstrap_cagr_p05_min
        and bootstrap_summary["bootstrap_sharpe_p05"] >= config.bootstrap_sharpe_p05_min
        and bootstrap_summary[f"bootstrap_prob_{config.target_metric}_positive"]
        >= config.prob_target_positive_min
    )
    statistical_pass = bool(
        bootstrap_pass
        and split["split_pass"]
        and looy["leave_one_year_out_pass"]
        and calendar["calendar_profit_pass"]
        and abs(float(metrics["mdd"])) <= config.max_drawdown_limit
        and psr_dsr["psr_dsr_pass"]
    )
    robust_pass = bool(
        data_quality.passed
        and statistical_pass
        and multiple["multiple_testing_pass"]
        and cost["cost_pass"]
    )
    correlation_pass = bool(benchmark["correlation_pass"])

    fail_reasons = []
    fail_reasons.extend(data_quality.fail_reasons)
    if not bootstrap_pass:
        fail_reasons.append("bootstrap_pass_false")
    if not split["split_pass"]:
        fail_reasons.append("split_pass_false")
    if not looy["leave_one_year_out_pass"]:
        fail_reasons.append("leave_one_year_out_pass_false")
    if not calendar["calendar_profit_pass"]:
        fail_reasons.append("calendar_profit_pass_false")
    if abs(float(metrics["mdd"])) > config.max_drawdown_limit:
        fail_reasons.append("max_drawdown_limit_exceeded")
    if not psr_dsr["psr_dsr_pass"]:
        fail_reasons.append("psr_dsr_pass_false")
    if not multiple["multiple_testing_pass"]:
        fail_reasons.append(multiple["multiple_testing_fail_reason"])
    if not cost["cost_pass"]:
        fail_reasons.append(cost["cost_fail_reason"])
    if not correlation_pass:
        fail_reasons.append("correlation_pass_false")

    result: dict[str, Any] = {
        "candidate_id": candidate_id,
        "run_id": run_id,
        "source_method": source_method,
        "frequency": frequency,
        "sample_role": sample_role,
        **metrics,
        **bootstrap_summary,
        **split,
        **looy,
        **calendar,
        **psr_dsr,
        **multiple,
        **cost,
        **benchmark,
        "data_quality_pass": bool(data_quality.passed),
        "data_quality_warnings": ";".join(data_quality.warnings),
        "bootstrap_pass": bool(bootstrap_pass),
        "statistical_pass": bool(statistical_pass),
        "robust_pass": bool(robust_pass),
        "duplicate_group_id": "",
        "duplicate_group_size": 1,
        "duplicate_representative": True,
        "portfolio_eligible": bool(robust_pass and correlation_pass),
        "fail_reasons": ";".join(reason for reason in fail_reasons if reason),
    }
    return UniversalRobustnessResult(
        result=result,
        bootstrap=bootstrap_df.assign(candidate_id=candidate_id),
        year_by_year=year_by_year.assign(candidate_id=candidate_id),
        data_quality={
            "candidate_id": candidate_id,
            "data_quality_pass": bool(data_quality.passed),
            "fail_reasons": ";".join(data_quality.fail_reasons),
            "warnings": ";".join(data_quality.warnings),
            **data_quality.metrics,
        },
    )


def run_universal_robustness_from_positions(
    positions: pd.DataFrame,
    prices: pd.DataFrame | pd.Series,
    config: UniversalRobustnessConfig | None = None,
) -> UniversalRobustnessResult:
    """Convert positions plus prices to returns, then run robustness."""

    config = config or UniversalRobustnessConfig()
    frame = positions.copy()
    price_series = _price_series(prices)
    price_returns = price_series.pct_change().reindex(pd.to_datetime(frame["timestamp"])).fillna(0.0)
    exposure = pd.to_numeric(frame.get("exposure", 1.0), errors="coerce").fillna(0.0)
    frame["strategy_return"] = exposure.to_numpy(dtype=float) * price_returns.to_numpy(dtype=float)
    if "exposure" not in frame.columns:
        frame["exposure"] = exposure
    return run_universal_robustness(frame, config)


def run_batch_universal_robustness(
    returns: pd.DataFrame,
    config: UniversalRobustnessConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the universal robustness test over many candidate return series."""

    config = config or UniversalRobustnessConfig()
    frame = _normalise_returns(returns)
    if frame.empty:
        empty = pd.DataFrame()
        return {
            "results": empty,
            "pass": empty,
            "methods": empty,
            "year_by_year": empty,
            "fail_reasons": empty,
            "duplicates": empty,
            "bootstrap": empty,
            "data_quality": empty,
        }

    result_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[pd.DataFrame] = []
    yby_rows: list[pd.DataFrame] = []
    quality_rows: list[dict[str, Any]] = []

    for _, group in frame.groupby("candidate_id", sort=False):
        report = run_universal_robustness(group, config)
        result_rows.append(report.result)
        bootstrap_rows.append(report.bootstrap)
        yby_rows.append(report.year_by_year)
        quality_rows.append(report.data_quality)

    results = pd.DataFrame(result_rows)
    results, duplicates = assign_duplicate_groups(results, frame, config)
    pass_results = results[results["robust_pass"].astype(bool)].copy() if not results.empty else pd.DataFrame()
    methods = _method_summary(results)
    fail_reasons = _fail_reason_table(results)
    bootstrap = pd.concat(bootstrap_rows, ignore_index=True) if bootstrap_rows else pd.DataFrame()
    year_by_year = pd.concat(yby_rows, ignore_index=True) if yby_rows else pd.DataFrame()
    data_quality = pd.DataFrame(quality_rows)
    return {
        "results": results,
        "pass": pass_results,
        "methods": methods,
        "year_by_year": year_by_year,
        "fail_reasons": fail_reasons,
        "duplicates": duplicates,
        "bootstrap": bootstrap,
        "data_quality": data_quality,
    }


def _normalise_returns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        data = data.sort_values(["candidate_id", "timestamp"] if "candidate_id" in data.columns else ["timestamp"])
    return data.reset_index(drop=True)


def _empty_result() -> dict[str, Any]:
    return {
        "candidate_id": "",
        "run_id": "",
        "source_method": "",
        "frequency": "",
        "sample_role": "",
        "data_quality_pass": False,
        "statistical_pass": False,
        "multiple_testing_pass": False,
        "cost_pass": False,
        "robust_pass": False,
        "portfolio_eligible": False,
        "fail_reasons": "empty_returns",
    }


def _sample_role_summary(series: pd.Series) -> str:
    values = [str(value) for value in series.dropna().unique()]
    return ",".join(values)


def _return_metrics(values: np.ndarray, ppy: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"cagr": 0.0, "mdd": 0.0, "calmar": 0.0, "sharpe": 0.0}
    equity = np.cumprod(1.0 + values)
    years = max(len(values) / float(ppy), 1e-9)
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0.0 else -1.0
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    mdd = float(np.min(drawdown))
    calmar = float(cagr / abs(mdd)) if mdd < -1e-12 else (float("inf") if cagr > 0.0 else 0.0)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sharpe = float(np.mean(values) / std * math.sqrt(float(ppy))) if std > 1e-12 else 0.0
    profit_periods_pct = float(np.mean(values > 0.0)) if len(values) else 0.0
    return {
        "cagr": cagr,
        "mdd": mdd,
        "calmar": calmar,
        "sharpe": sharpe,
        "profit_periods_pct": profit_periods_pct,
        "periods": int(len(values)),
    }


def _bootstrap_metrics(
    values: np.ndarray,
    frequency: str,
    config: UniversalRobustnessConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    ppy = config.periods_per_year(frequency)
    block = max(1, int(config.block_length(frequency)))
    samples = int(config.bootstrap_samples)
    rng = np.random.default_rng(int(config.bootstrap_seed))
    rows: list[dict[str, float | int]] = []
    if len(values) == 0:
        values = np.array([0.0])
    for sample_index in range(samples):
        sample = _block_sample(values, block, rng)
        metrics = _return_metrics(sample, ppy)
        rows.append(
            {
                "bootstrap_sample": sample_index,
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "calmar": metrics["calmar"],
            }
        )
    df = pd.DataFrame(rows)
    target = str(config.target_metric)
    if target not in df.columns:
        target = "cagr"
    summary = {
        "bootstrap_cagr_p05": float(df["cagr"].quantile(0.05)),
        "bootstrap_cagr_p50": float(df["cagr"].quantile(0.50)),
        "bootstrap_cagr_p95": float(df["cagr"].quantile(0.95)),
        "bootstrap_sharpe_p05": float(df["sharpe"].quantile(0.05)),
        "bootstrap_sharpe_p50": float(df["sharpe"].quantile(0.50)),
        "bootstrap_sharpe_p95": float(df["sharpe"].quantile(0.95)),
        "bootstrap_calmar_p05": float(df["calmar"].replace([np.inf, -np.inf], np.nan).quantile(0.05)),
        "bootstrap_prob_cagr_positive": float(np.mean(df["cagr"] > 0.0)),
        f"bootstrap_prob_{target}_positive": float(np.mean(df[target] > 0.0)),
    }
    if f"bootstrap_prob_{config.target_metric}_positive" not in summary:
        summary[f"bootstrap_prob_{config.target_metric}_positive"] = summary[f"bootstrap_prob_{target}_positive"]
    return df, summary


def _block_sample(values: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    if len(values) <= block:
        return values.copy()
    chunks: list[np.ndarray] = []
    while sum(len(chunk) for chunk in chunks) < len(values):
        start = int(rng.integers(0, len(values)))
        indices = (np.arange(start, start + block) % len(values)).astype(int)
        chunks.append(values[indices])
    return np.concatenate(chunks)[: len(values)]


def _split_checks(values: np.ndarray, ppy: int, config: UniversalRobustnessConfig) -> dict[str, float | bool]:
    midpoint = len(values) // 2
    first = _return_metrics(values[:midpoint], ppy)
    second = _return_metrics(values[midpoint:], ppy)
    passed = first["cagr"] >= config.half_1_cagr_min and second["cagr"] >= config.half_2_cagr_min
    return {
        "half_1_cagr": first["cagr"],
        "half_2_cagr": second["cagr"],
        "split_pass": bool(passed),
    }


def _leave_one_year_out(
    frame: pd.DataFrame,
    ppy: int,
    config: UniversalRobustnessConfig,
) -> dict[str, float | int | bool]:
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    years = sorted(timestamps.dt.year.dropna().unique())
    if len(years) < 3:
        return {
            "leave_one_year_out_pass": True,
            "leave_one_year_out_min_cagr": np.nan,
            "leave_one_year_out_years": int(len(years)),
        }
    cagr_values: list[float] = []
    returns = pd.to_numeric(frame["strategy_return"], errors="coerce")
    for year in years:
        subset = returns[timestamps.dt.year != year].to_numpy(dtype=float)
        cagr_values.append(_return_metrics(subset, ppy)["cagr"])
    min_cagr = float(np.min(cagr_values)) if cagr_values else np.nan
    return {
        "leave_one_year_out_pass": bool(min_cagr >= config.leave_one_year_out_min_cagr),
        "leave_one_year_out_min_cagr": min_cagr,
        "leave_one_year_out_years": int(len(years)),
    }


def _calendar_profit_checks(
    frame: pd.DataFrame,
    config: UniversalRobustnessConfig,
) -> dict[str, float | bool]:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["strategy_return"] = pd.to_numeric(data["strategy_return"], errors="coerce").fillna(0.0)
    monthly = data.groupby(data["timestamp"].dt.to_period("M"))["strategy_return"].apply(
        lambda values: float((1.0 + values).prod() - 1.0)
    )
    yearly = data.groupby(data["timestamp"].dt.year)["strategy_return"].apply(
        lambda values: float((1.0 + values).prod() - 1.0)
    )
    profit_months_pct = float((monthly > 0.0).mean()) if len(monthly) else 0.0
    profit_years_pct = float((yearly > 0.0).mean()) if len(yearly) else 0.0
    passed = (
        profit_months_pct >= float(config.min_profit_months_pct)
        and profit_years_pct >= float(config.min_profit_years_pct)
    )
    return {
        "profit_months_pct": profit_months_pct,
        "profit_years_pct": profit_years_pct,
        "calendar_profit_pass": bool(passed),
    }


def _multiple_testing_check(
    frame: pd.DataFrame,
    source_method: str,
    config: UniversalRobustnessConfig,
) -> dict[str, Any]:
    generated = source_method in set(config.generated_methods)
    n_trials_value = frame["n_trials"].dropna().iloc[0] if "n_trials" in frame.columns and frame["n_trials"].notna().any() else None
    if n_trials_value is None:
        if generated:
            return {
                "n_trials": np.nan,
                "multiple_testing_pass": False,
                "multiple_testing_fail_reason": "missing_n_trials_generated_strategy",
            }
        return {"n_trials": 1, "multiple_testing_pass": True, "multiple_testing_fail_reason": ""}
    n_trials = int(float(n_trials_value))
    if n_trials < 1:
        return {"n_trials": n_trials, "multiple_testing_pass": False, "multiple_testing_fail_reason": "n_trials_lt_1"}
    return {"n_trials": n_trials, "multiple_testing_pass": True, "multiple_testing_fail_reason": ""}


def _psr_dsr_check(
    values: np.ndarray,
    ppy: int,
    frame: pd.DataFrame,
    config: UniversalRobustnessConfig,
    n_trials: Any,
) -> dict[str, float | bool]:
    metrics = _return_metrics(values, ppy)
    try:
        trials = int(float(n_trials))
    except (TypeError, ValueError):
        return {"psr": np.nan, "dsr": np.nan, "psr_dsr_pass": False}
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3:
        return {"psr": np.nan, "dsr": np.nan, "psr_dsr_pass": False}
    skew = float(stats.skew(finite, bias=False, nan_policy="omit"))
    kurtosis = float(stats.kurtosis(finite, fisher=False, bias=False, nan_policy="omit"))
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurtosis):
        kurtosis = 3.0
    report = deflated_sharpe_annualized(
        metrics["sharpe"],
        n_trials=trials,
        n_periods=len(finite),
        ppy=ppy,
        skew=skew,
        kurtosis=kurtosis,
        min_dsr=config.dsr_min,
        min_psr=config.psr_min,
    )
    return {
        "psr": float(report.psr_vs_zero),
        "dsr": float(report.dsr),
        "psr_dsr_pass": bool(report.passed),
    }


def _cost_check(
    frame: pd.DataFrame,
    ppy: int,
    config: UniversalRobustnessConfig,
) -> dict[str, float | bool | str]:
    returns_are_net = _truthy(frame["returns_are_net"].iloc[0])
    values = pd.to_numeric(frame["strategy_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base_score = _metric_score(values, ppy, config.target_metric)
    if returns_are_net:
        return {
            "returns_are_net": True,
            "cost_pass": True,
            "cost_score_retention": 1.0,
            "cost_fail_reason": "",
        }
    if "turnover" not in frame.columns or frame["turnover"].isna().all():
        return {
            "returns_are_net": False,
            "cost_pass": False,
            "cost_score_retention": np.nan,
            "cost_fail_reason": "gross_returns_missing_turnover",
        }
    turnover = pd.to_numeric(frame["turnover"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cost_rate = (float(config.slippage_bps) + float(config.commission_bps)) / 10000.0
    stressed = values - turnover * cost_rate * float(config.cost_stress_multiplier)
    stressed_score = _metric_score(stressed, ppy, config.target_metric)
    retention = stressed_score / base_score if base_score > 1e-12 else -np.inf
    passed = bool(retention >= config.cost_stress_score_retention_min)
    return {
        "returns_are_net": False,
        "cost_pass": passed,
        "cost_score_retention": float(retention),
        "cost_fail_reason": "" if passed else "cost_stress_retention_too_low",
    }


def _benchmark_checks(frame: pd.DataFrame, config: UniversalRobustnessConfig) -> dict[str, float | bool]:
    if "benchmark_return" not in frame.columns or frame["benchmark_return"].isna().all():
        return {
            "benchmark_corr": np.nan,
            "benchmark_beta": np.nan,
            "correlation_pass": True,
        }
    strategy = pd.to_numeric(frame["strategy_return"], errors="coerce")
    benchmark = pd.to_numeric(frame["benchmark_return"], errors="coerce")
    aligned = pd.concat([strategy, benchmark], axis=1).dropna()
    if len(aligned) < 3:
        return {"benchmark_corr": np.nan, "benchmark_beta": np.nan, "correlation_pass": True}
    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    variance = float(np.var(aligned.iloc[:, 1].to_numpy(dtype=float), ddof=1))
    covariance = float(np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)[0, 1])
    beta = covariance / variance if variance > 1e-12 else np.nan
    passed = bool(abs(corr) <= config.benchmark_corr_max and (not np.isfinite(beta) or abs(beta) <= config.beta_abs_max))
    return {"benchmark_corr": corr, "benchmark_beta": float(beta), "correlation_pass": passed}


def _year_by_year(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["year"] = data["timestamp"].dt.year
    rows: list[dict[str, Any]] = []
    for (year, role), group in data.groupby(["year", "sample_role"], dropna=False):
        returns = pd.to_numeric(group["strategy_return"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "year": int(year),
                "sample_role": str(role),
                "periods": int(len(group)),
                "strategy_return": float((1.0 + returns).prod() - 1.0),
            }
        )
    return pd.DataFrame(rows)


def _metric_score(values: np.ndarray, ppy: int, metric: str) -> float:
    metrics = _return_metrics(values, ppy)
    value = float(metrics.get(metric, metrics["cagr"]))
    if not np.isfinite(value):
        return 0.0
    return value


def _price_series(prices: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(prices, pd.Series):
        return pd.to_numeric(prices, errors="coerce")
    for column in ("close", "adj_close", "price"):
        if column in prices.columns:
            return pd.to_numeric(prices[column], errors="coerce")
    if len(prices.columns) == 1:
        return pd.to_numeric(prices.iloc[:, 0], errors="coerce")
    raise ValueError("prices must be a Series or contain close/adj_close/price")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _method_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for method, group in results.groupby("source_method", dropna=False):
        rows.append(
            {
                "source_method": str(method),
                "candidates": int(len(group)),
                "robust_pass": int(group["robust_pass"].fillna(False).sum()),
                "portfolio_eligible": int(group["portfolio_eligible"].fillna(False).sum()),
                "best_cagr": float(group["cagr"].max()) if "cagr" in group else np.nan,
                "best_sharpe": float(group["sharpe"].max()) if "sharpe" in group else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _fail_reason_table(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in results.iterrows():
        for reason in str(row.get("fail_reasons", "") or "").split(";"):
            if reason:
                rows.append({"candidate_id": row["candidate_id"], "fail_reason": reason})
    return pd.DataFrame(rows, columns=["candidate_id", "fail_reason"])
