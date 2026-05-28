from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aurora.validation.robustness_config import UniversalRobustnessConfig


REQUIRED_COLUMNS = (
    "candidate_id",
    "run_id",
    "source_method",
    "timestamp",
    "frequency",
    "strategy_return",
    "returns_are_net",
    "sample_role",
)


@dataclass(frozen=True)
class DataQualityResult:
    passed: bool
    fail_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: dict[str, float | int | str | bool]


def validate_return_frame(
    frame: pd.DataFrame,
    config: UniversalRobustnessConfig,
) -> DataQualityResult:
    """Validate one candidate return series before statistical tests."""

    fail_reasons: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, float | int | str | bool] = {}

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        return DataQualityResult(
            passed=False,
            fail_reasons=tuple(f"missing_column:{column}" for column in missing),
            warnings=(),
            metrics={"rows": int(len(frame))},
        )

    if frame.empty:
        return DataQualityResult(False, ("empty_returns",), (), {"rows": 0})

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any():
        fail_reasons.append("invalid_timestamp")
    if timestamps.duplicated().any():
        fail_reasons.append("duplicate_timestamp")
    if not timestamps.is_monotonic_increasing:
        fail_reasons.append("timestamp_not_sorted")

    frequency = str(frame["frequency"].dropna().iloc[0]).lower()
    if frequency not in config.min_periods_by_frequency:
        fail_reasons.append(f"unknown_frequency:{frequency}")

    returns = pd.to_numeric(frame["strategy_return"], errors="coerce")
    finite_mask = np.isfinite(returns.to_numpy(dtype=float, na_value=np.nan))
    if returns.isna().any():
        fail_reasons.append("nan_return")
    if not bool(finite_mask.all()):
        fail_reasons.append("infinite_return")
    if (returns < -1.0).any():
        fail_reasons.append("return_below_minus_100pct")

    values = returns.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    rows = int(len(values))
    metrics["rows"] = rows
    metrics["frequency"] = frequency

    if frequency in config.min_periods_by_frequency:
        min_periods = int(config.min_periods_by_frequency[frequency])
        metrics["min_periods_required"] = min_periods
        if rows < min_periods:
            fail_reasons.append(f"too_few_periods:{rows}<{min_periods}")

        active_periods = int(np.sum(np.abs(values) > 1e-12))
        min_active = int(config.min_active_periods_by_frequency[frequency])
        metrics["active_periods"] = active_periods
        metrics["min_active_periods_required"] = min_active
        if active_periods < min_active:
            fail_reasons.append(f"too_few_active_periods:{active_periods}<{min_active}")

    if rows > 1:
        std = float(np.std(values, ddof=1))
        metrics["std"] = std
        if std <= 1e-12:
            fail_reasons.append("zero_return_std")

    if rows and bool(np.all(values > 0.0)):
        warnings.append("all_returns_positive")

    positive = values[values > 0.0]
    total_positive = float(np.sum(positive)) if len(positive) else 0.0
    max_positive = float(np.max(positive)) if len(positive) else 0.0
    contribution = max_positive / total_positive if total_positive > 0.0 else 0.0
    metrics["max_single_period_profit_contribution"] = float(contribution)
    if contribution > float(config.max_single_period_profit_contribution):
        fail_reasons.append(
            "single_period_profit_concentration:"
            f"{contribution:.6f}>{config.max_single_period_profit_contribution:.6f}"
        )

    return DataQualityResult(
        passed=not fail_reasons,
        fail_reasons=tuple(fail_reasons),
        warnings=tuple(warnings),
        metrics=metrics,
    )
